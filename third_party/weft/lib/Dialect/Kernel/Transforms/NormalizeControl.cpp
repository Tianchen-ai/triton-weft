#include "Weft/Dialect/Kernel/Transforms/SourceSpecialization.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Analysis/IntegerFacts.h"
#include "Weft/Analysis/IndexRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace {
class FoldScalarInteger : public RewritePattern {
public:
  explicit FoldScalarInteger(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *operation,
                               PatternRewriter &rewriter) const override {
    namespace k = weft::kernel;
    if (!isa<k::CastOp, k::WidenOp, k::BinaryOp>(operation) ||
        operation->getNumResults() != 1) return failure();
    auto result = operation->getResult(0);
    auto integer = dyn_cast<IntegerType>(result.getType());
    bool index = result.getType().isIndex();
    if ((!integer && !index) || (integer && integer.getWidth() == 1)) return failure();
    auto interpretation = index ? IntegerType::get(getContext(), 64, IntegerType::Signed) : integer;
    if (index) {
      auto outer = dyn_cast<k::CastOp>(operation);
      auto inner = outer ? outer.getInput().getDefiningOp<k::CastOp>() : k::CastOp();
      auto carrier = inner ? dyn_cast<IntegerType>(inner.getResult().getType()) : IntegerType();
      if (inner && carrier && inner.getInput().getType().isIndex()) {
        auto source = weft::analysis::integerRange(inner.getInput(), interpretation);
        // Ordinary frontend loops may expose an I32 induction variable while
        // View addressing uses index. Only remove that round trip when the
        // source bounds prove that the intermediate cast cannot lose bits.
        if (source && !source->minimum.isNegative() && source->fits(carrier)) {
          rewriter.replaceOp(operation, inner.getInput());
          return success();
        }
      }
    }
    auto range = weft::analysis::integerRange(result, interpretation);
    if (!range || range->minimum != range->maximum) return failure();
    if (index) {
      if (range->minimum.getSignificantBits() > 64) return failure();
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(operation,
          range->minimum.extOrTrunc(64).getSExtValue());
    } else {
      auto literal = IntegerAttr::get(integer, range->minimum.extOrTrunc(integer.getWidth()));
      rewriter.replaceOpWithNewOp<k::ConstantOp>(operation, integer, literal);
    }
    return success();
  }
};

class FoldScalarIntegerCompare : public OpRewritePattern<weft::kernel::CompareOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::CompareOp compare,
                               PatternRewriter &rewriter) const override {
    auto type = compare.getLhs().getType();
    auto integer = dyn_cast<IntegerType>(type);
    if ((!integer && !type.isIndex()) ||
        compare.getRhs().getType() != type ||
        !compare.getResult().getType().isInteger(1)) return failure();
    auto interpretation = type.isIndex()
        ? IntegerType::get(getContext(), 64, IntegerType::Signed) : integer;
    auto left = weft::analysis::integerRange(compare.getLhs(), interpretation);
    auto right = weft::analysis::integerRange(compare.getRhs(), interpretation);
    if (!left || !right || left->minimum != left->maximum ||
        right->minimum != right->maximum) return failure();
    int order = llvm::APSInt::compareValues(left->minimum, right->minimum);
    auto predicate = compare.getPredicate();
    bool result;
    if (predicate == "eq") result = order == 0;
    else if (predicate == "ne") result = order != 0;
    else if (predicate == "lt") result = order < 0;
    else if (predicate == "le") result = order <= 0;
    else if (predicate == "gt") result = order > 0;
    else if (predicate == "ge") result = order >= 0;
    else return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(compare, result, 1);
    return success();
  }
};

class FoldConstantIf : public OpRewritePattern<scf::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp operation,
                               PatternRewriter &rewriter) const override {
    auto constant = operation.getCondition().getDefiningOp<arith::ConstantOp>();
    auto condition = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
    if (!condition) return failure();
    Region &selected = condition.getValue().isZero()
        ? operation.getElseRegion() : operation.getThenRegion();
    if (selected.empty()) {
      if (operation.getNumResults()) return failure();
      rewriter.eraseOp(operation);
      return success();
    }
    if (!llvm::hasSingleElement(selected)) return failure();
    auto yield = dyn_cast<scf::YieldOp>(selected.front().getTerminator());
    if (!yield) return failure();
    SmallVector<Value> results(yield.getOperands());
    rewriter.eraseOp(yield);
    rewriter.inlineBlockBefore(&selected.front(), operation);
    rewriter.replaceOp(operation, results);
    return success();
  }
};

class NormalizeScalarSelect : public OpRewritePattern<arith::SelectOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::SelectOp select,
                               PatternRewriter &rewriter) const override {
    if (!select.getCondition().getType().isInteger(1) ||
        !isa<IntegerType, IndexType, FloatType>(select.getType()))
      return failure();
    // SCF's upstream canonicalization can replace a pure scalar if/yield with
    // arith.select. Keep that simplification inside the Canonical value model.
    rewriter.replaceOpWithNewOp<weft::kernel::SelectOp>(
        select, select.getType(), select.getCondition(), select.getTrueValue(),
        select.getFalseValue());
    return success();
  }
};

class FoldIdentitySlice : public OpRewritePattern<weft::kernel::SliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::SliceOp slice,
                               PatternRewriter &rewriter) const override {
    if (slice.getBase().getType() != slice.getResult().getType() ||
        !slice.getIndices().empty() ||
        !llvm::all_of(slice.getSelectors(), [](Attribute selector) {
          return cast<StringAttr>(selector).getValue() == "all";
        })) return failure();
    rewriter.replaceOp(slice, slice.getBase());
    return success();
  }
};

bool hasEncodedPayloadAxis(Value value) {
  auto field = value.getDefiningOp<weft::kernel::FieldOp>();
  if (!field) return false;
  auto owner = dyn_cast<weft::kernel::ValueType>(field.getOwner().getType());
  auto result = dyn_cast<weft::kernel::ValueType>(value.getType());
  return owner && result && !owner.getAxisIds().empty() &&
      llvm::is_contained(result.getAxisIds().asArrayRef(), owner.getAxisIds().asArrayRef().back());
}

class NormalizeBoundedGather : public OpRewritePattern<weft::kernel::ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::ExtractOp extract,
                               PatternRewriter &rewriter) const override {
    auto table = dyn_cast<weft::kernel::ValueType>(extract.getInput().getType());
    if (!table || table.getShape().size() != 1 || table.getShape()[0] <= 0 ||
        extract.getIndices().size() != 1 || extract.getSelectors().size() != 1 ||
        cast<StringAttr>(extract.getSelectors()[0]).getValue() != "gather")
      return failure();
    // Retain grouped payload projections until target memory planning. A
    // scalar-per-record field instead has only outer record axes: an identity
    // gather there is an ordinary numeric axis rename, not a payload window.
    if (hasEncodedPayloadAxis(extract.getInput()))
      return failure();
    Value indices = extract.getIndices()[0];
    if (!weft::analysis::hasUnsignedBound(indices, table.getShape()[0]))
      return failure();
    // A fully bounded one-axis numerical gather is the same table operation
    // as an explicit lookup. Preserve both SSA snapshots and the index domain;
    // no load, alias boundary or author traversal is introduced here.
    rewriter.replaceOpWithNewOp<weft::kernel::LookupOp>(extract,
        extract.getResult().getType(), extract.getInput(), indices,
        rewriter.getStringAttr("in_bounds"));
    return success();
  }
};

class FoldIdentityLookup : public OpRewritePattern<weft::kernel::LookupOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::LookupOp lookup,
                               PatternRewriter &rewriter) const override {
    namespace k = weft::kernel;
    auto table = dyn_cast<k::ValueType>(lookup.getTable().getType());
    auto indices = dyn_cast<k::ValueType>(lookup.getIndices().getType());
    auto result = dyn_cast<k::ValueType>(lookup.getResult().getType());
    if (!table || !indices || !result || table.getShape().size() != 1 ||
        table.getShape()[0] <= 0 || result.getShape() != indices.getShape() ||
        result.getElementType() != table.getElementType())
      return failure();
    if (hasEncodedPayloadAxis(lookup.getTable()))
      return failure();
    int64_t elements = 1;
    for (int64_t extent : indices.getShape().asArrayRef()) {
      if (extent <= 0 || elements > table.getShape()[0] / extent) return failure();
      elements *= extent;
    }
    if (elements != table.getShape()[0] ||
        !weft::analysis::hasRowMajorCoordinates(lookup.getIndices())) return failure();
    // N consecutive, non-wrapping indices within [0, N) select the original
    // SSA coordinates. Only their axis names can differ; the read epoch and
    // any state/stage birth remain on the original value.
    if (table == result) rewriter.replaceOp(lookup, lookup.getTable());
    else rewriter.replaceOpWithNewOp<k::ReshapeOp>(lookup, result,
        lookup.getTable(), table.getAxisIds());
    return success();
  }
};

class FoldIdentityTileGather : public OpRewritePattern<weft::kernel::ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::ExtractOp extract,
                                PatternRewriter &rewriter) const override {
    namespace k = weft::kernel;
    auto input = dyn_cast<k::ValueType>(extract.getInput().getType());
    auto output = dyn_cast<k::ValueType>(extract.getResult().getType());
    if (!input || !output || input.getShape().size() < 2 ||
        input.getShape() != output.getShape() ||
        input.getElementType() != output.getElementType() ||
        extract.getSelectors().size() != input.getShape().size())
      return failure();
    // Packed payload projections retain their record/layer supply until
    // target memory planning. Whole-byte metadata and ordinary numerical
    // tiles can express an identity gather as a coordinate-only rename.
    if (extract.getInput().getDefiningOp<k::FieldOp>())
      if (auto integer = dyn_cast<IntegerType>(input.getElementType());
          integer && integer.getWidth() < 8)
        return failure();
    size_t next = 0;
    for (auto [axis, selector] : llvm::enumerate(extract.getSelectors())) {
      StringRef kind = cast<StringAttr>(selector).getValue();
      if (kind == "all") {
        if (input.getAxisIds()[axis] != output.getAxisIds()[axis]) return failure();
        continue;
      }
      if (kind != "gather" || next == extract.getIndices().size()) return failure();
      Value index = extract.getIndices()[next++];
      auto type = dyn_cast<k::ValueType>(index.getType());
      int64_t extent = input.getShape()[axis];
      if (!type || type.getShape().size() != 1 || extent <= 0 ||
          type.getShape()[0] != extent ||
          type.getAxisIds()[0] != output.getAxisIds()[axis] ||
          !weft::analysis::hasUnsignedBound(index, extent) ||
          !weft::analysis::hasRowMajorCoordinates(index))
        return failure();
    }
    if (next != extract.getIndices().size()) return failure();
    rewriter.replaceOpWithNewOp<k::ReshapeOp>(extract, output,
        extract.getInput(), input.getAxisIds());
    return success();
  }
};

class EraseUnusedEmpty : public OpRewritePattern<weft::kernel::EmptyOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::EmptyOp empty,
                               PatternRewriter &rewriter) const override {
    // This uninitialized numerical container has no address, returned storage,
    // or state/stage identity. A complete SSA assignment can remove its only
    // user; retaining the unused allocation would introduce a false effect.
    if (!empty.getResult().use_empty()) return failure();
    rewriter.eraseOp(empty);
    return success();
  }
};

class FoldStaticSubviewOffsets : public OpRewritePattern<weft::kernel::SubviewOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::SubviewOp subview,
                               PatternRewriter &rewriter) const override {
    SmallVector<int64_t> offsets(subview.getOffsets());
    SmallVector<Value> dynamic;
    unsigned cursor = 0;
    bool changed = false;
    for (int64_t &offset : offsets) {
      if (offset != -1) continue;
      Value bound = subview.getDynamicBounds()[cursor++];
      auto range = weft::analysis::integerRange(bound,
          IntegerType::get(getContext(), 64, IntegerType::Signed));
      if (range && range->minimum == range->maximum && range->minimum >= 0 &&
          range->minimum.getSignificantBits() <= 64) {
        offset = range->minimum.extOrTrunc(64).getSExtValue();
        changed = true;
      } else dynamic.push_back(bound);
    }
    if (!changed) return failure();
    llvm::append_range(dynamic, subview.getDynamicBounds().drop_front(cursor));
    rewriter.modifyOpInPlace(subview, [&] {
      subview.setOffsetsAttr(rewriter.getDenseI64ArrayAttr(offsets));
      subview.getDynamicBoundsMutable().assign(dynamic);
    });
    return success();
  }
};

class FoldDescriptorExtent : public OpRewritePattern<weft::kernel::ExtentOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::ExtentOp extent,
                               PatternRewriter &rewriter) const override {
    namespace k = weft::kernel;
    Value base = extent.getBase();
    const unsigned axis = extent.getAxis();
    if (auto subview = base.getDefiningOp<k::SubviewOp>()) {
      int64_t size = subview.getExtents()[axis];
      if (size >= 0) {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(extent, size);
      } else {
        unsigned dynamic = 0;
        for (int64_t offset : subview.getOffsets()) dynamic += offset == -1;
        for (unsigned i = 0; i < axis; ++i) dynamic += subview.getExtents()[i] == -1;
        rewriter.replaceOp(extent, subview.getDynamicBounds()[dynamic]);
      }
      return success();
    }
    if (auto bind = base.getDefiningOp<k::BindViewOp>()) {
      rewriter.replaceOp(extent, bind.getExtents()[axis]);
      return success();
    }
    // A View carries its exact declared extent. A Level slice instead carries
    // a nominal capacity; retain its query until the active domain is known.
    if (auto view = dyn_cast<k::ViewType>(base.getType())) {
      int64_t size = view.getShape()[axis];
      if (size < 0) return failure();
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(extent, size);
      return success();
    }
    return failure();
  }
};

bool isZeroIndex(Value value) {
  if (auto literal = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(literal.getValue());
    return integer && integer.getValue().isZero();
  }
  if (auto literal = value.getDefiningOp<weft::kernel::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(literal.getValue());
    return integer && integer.getValue().isZero();
  }
  if (auto cast = value.getDefiningOp<weft::kernel::CastOp>())
    return isZeroIndex(cast.getInput());
  return false;
}

bool isFullBroadcastCondition(Value value) {
  namespace k = weft::kernel;
  Attribute literal;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    literal = constant.getValue();
  else if (auto constant = value.getDefiningOp<k::ConstantOp>())
    literal = constant.getValue();
  if (literal) {
    auto integer = dyn_cast<IntegerAttr>(literal);
    return integer && integer.getType().isInteger(1) && integer.getValue().isOne();
  }
  if (auto compare = value.getDefiningOp<k::CompareOp>()) {
    Value coordinate = compare.getLhs();
    while (true) {
      if (auto cast = coordinate.getDefiningOp<k::CastOp>())
        coordinate = cast.getInput();
      else if (auto widen = coordinate.getDefiningOp<k::WidenOp>())
        coordinate = widen.getInput();
      else break;
    }
    return compare.getPredicate() == "eq" && compare.getLhs() == compare.getRhs() &&
        coordinate.getDefiningOp<k::IotaOp>();
  }
  if (auto binary = value.getDefiningOp<k::BinaryOp>())
    return binary.getKind() == "and" && isFullBroadcastCondition(binary.getLhs()) &&
        isFullBroadcastCondition(binary.getRhs());
  if (auto reshape = value.getDefiningOp<k::ReshapeOp>())
    return isFullBroadcastCondition(reshape.getInput());
  if (auto select = value.getDefiningOp<k::SelectOp>())
    return select.getTrueValue() == select.getFalseValue() &&
        isFullBroadcastCondition(select.getCondition()) &&
        isFullBroadcastCondition(select.getTrueValue());
  return false;
}

class FoldFullSelect : public OpRewritePattern<weft::kernel::SelectOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::SelectOp select,
                               PatternRewriter &rewriter) const override {
    if (!isFullBroadcastCondition(select.getCondition())) return failure();
    // A complete coordinate assignment can already have the destination's
    // exact type. No broadcast or validity predicate remains to materialize.
    if (select.getTrueValue().getType() == select.getResult().getType()) {
      rewriter.replaceOp(select, select.getTrueValue());
      return success();
    }
    namespace k = weft::kernel;
    auto input = dyn_cast<k::ValueType>(select.getTrueValue().getType());
    auto result = dyn_cast<k::ValueType>(select.getResult().getType());
    if (!input || !result || input.getElementType() != result.getElementType())
      return failure();
    SmallVector<std::pair<int64_t, int64_t>> sourceAxes, resultAxes;
    auto collect = [](k::ValueType type, auto &axes) {
      for (auto [extent, axis] : llvm::zip(type.getShape().asArrayRef(), type.getAxisIds().asArrayRef())) {
        if (extent <= 0) return false;
        if (extent != 1) axes.emplace_back(axis, extent);
      }
      return true;
    };
    if (!collect(input, sourceAxes) || !collect(result, resultAxes) ||
        sourceAxes != resultAxes) return failure();
    // Adding or removing singleton positional axes does not replicate payload.
    rewriter.replaceOpWithNewOp<k::ReshapeOp>(select, result,
        select.getTrueValue(), input.getAxisIds());
    return success();
  }
};

class FoldScalarProjection : public OpRewritePattern<weft::kernel::ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::ExtractOp extract,
                               PatternRewriter &rewriter) const override {
    namespace k = weft::kernel;
    auto type = dyn_cast<k::ValueType>(extract.getInput().getType());
    if (!type || isa<k::ValueType>(extract.getResult().getType()) ||
        extract.getSelectors().size() != type.getShape().size() ||
        extract.getIndices().size() != type.getShape().size() ||
        llvm::any_of(type.getShape().asArrayRef(), [](int64_t extent) { return extent <= 0; }) ||
        llvm::any_of(extract.getSelectors(), [](Attribute selector) {
          return cast<StringAttr>(selector).getValue() != "index";
        }) || !llvm::all_of(extract.getIndices(), isZeroIndex))
      return failure();
    Value input = extract.getInput();
    while (auto reshape = input.getDefiningOp<k::ReshapeOp>()) {
      if (reshape.getInput().getType().getShape() != reshape.getResult().getType().getShape() ||
          !llvm::equal(reshape.getOrder(), reshape.getInput().getType().getAxisIds().asArrayRef()))
        break;
      input = reshape.getInput();
    }
    Value scalar;
    if (auto select = input.getDefiningOp<k::SelectOp>();
        select && select.getTrueValue() == select.getFalseValue() &&
        isFullBroadcastCondition(select.getCondition()))
      scalar = select.getTrueValue();
    else if (auto update = input.getDefiningOp<k::UpdateOp>();
             update && update.getSelectors() == extract.getSelectors() &&
             update.getIndices().size() == extract.getIndices().size() &&
             llvm::all_of(update.getIndices(), isZeroIndex))
      scalar = update.getValue();
    if (!scalar || scalar.getType() != extract.getResult().getType())
      return failure();
    rewriter.replaceOp(extract, scalar);
    return success();
  }
};

class PushCoordinateRename : public OpRewritePattern<weft::kernel::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::ReshapeOp reshape,
                               PatternRewriter &rewriter) const override {
    namespace k = weft::kernel;
    auto input = reshape.getInput().getType();
    auto output = reshape.getResult().getType();
    if (input == output && llvm::equal(reshape.getOrder(), input.getAxisIds().asArrayRef())) {
      rewriter.replaceOp(reshape, reshape.getInput());
      return success();
    }
    if (input.getShape() != output.getShape() ||
        !llvm::equal(reshape.getOrder(), input.getAxisIds().asArrayRef()))
      return failure();
    // A positional tile copy can rename the same coordinates more than once
    // (for example, through an ordinary helper's destination). Compose those
    // identities before pushing the final rename into a numerical producer.
    // A transpose or change of shape still retains its explicit mapping.
    if (auto previous = reshape.getInput().getDefiningOp<k::ReshapeOp>()) {
      auto source = previous.getInput().getType();
      if (source.getShape() == input.getShape() &&
          llvm::equal(previous.getOrder(), source.getAxisIds().asArrayRef())) {
        rewriter.replaceOpWithNewOp<k::ReshapeOp>(reshape, output,
            previous.getInput(), previous.getOrderAttr());
        if (previous->use_empty()) rewriter.eraseOp(previous);
        return success();
      }
    }
    Operation *producer = reshape.getInput().getDefiningOp();
    if (!producer) return failure();
    bool elementwise = isa<k::IotaOp, k::UnaryOp, k::BinaryOp, k::CompareOp,
                           k::SelectOp, k::CastOp, k::BitcastOp, k::WidenOp, k::NarrowOp>(producer);
    auto extract = dyn_cast<k::ExtractOp>(producer);
    auto lookup = dyn_cast<k::LookupOp>(producer);
    if (!elementwise && !extract && !lookup) return failure();
    auto coordinateExpression = [&](auto &&self, Value value, unsigned depth) -> bool {
      auto type = dyn_cast<k::ValueType>(value.getType());
      if (!type || type.getShape().empty()) return true;
      if (depth > 16 || !isa<IntegerType>(type.getElementType())) return false;
      Operation *definition = value.getDefiningOp();
      if (!definition) return false;
      if (isa<k::IotaOp, k::ConstantOp>(definition)) return true;
      if (!isa<k::BinaryOp, k::CompareOp, k::SelectOp, k::CastOp,
               k::WidenOp, k::NarrowOp>(definition)) return false;
      return llvm::all_of(definition->getOperands(), [&](Value operand) {
        return self(self, operand, depth + 1);
      });
    };
    // Coordinate arithmetic often supplies several fields of one saved record.
    // Retain that shared relation across a positional rename rather than
    // materializing the old coordinate array. Reuse existing operand renames
    // below; reads and numerical payload graphs still require one use.
    if (!reshape.getInput().hasOneUse() && !extract &&
        !coordinateExpression(coordinateExpression, reshape.getInput(), 0))
      return failure();
    // An all/domain selector also carries the owner's coordinate identity.
    // Keep those boundaries explicit; only gather result coordinates commute.
    if (extract && llvm::any_of(extract.getSelectors(), [](Attribute attr) {
          return cast<StringAttr>(attr).getValue() != "gather" &&
                 cast<StringAttr>(attr).getValue() != "index";
        })) return failure();
    llvm::DenseMap<int64_t, int64_t> mapping;
    for (auto [oldAxis, newAxis] : llvm::zip(input.getAxisIds().asArrayRef(), output.getAxisIds().asArrayRef()))
      mapping[oldAxis] = newAxis;
    for (auto [position, operand] : llvm::enumerate(producer->getOperands())) {
      if ((extract || lookup) && position == 0) continue;
      if (auto type = dyn_cast<k::ValueType>(operand.getType()))
        for (int64_t axis : type.getAxisIds().asArrayRef())
          if (!mapping.contains(axis)) return failure();
    }
    SmallVector<Value> operands;
    for (auto [position, operand] : llvm::enumerate(producer->getOperands())) {
      auto type = dyn_cast<k::ValueType>(operand.getType());
      if (!type || ((extract || lookup) && position == 0)) {
        operands.push_back(operand);
        continue;
      }
      SmallVector<int64_t> axes;
      for (int64_t axis : type.getAxisIds().asArrayRef()) {
        auto it = mapping.find(axis);
        axes.push_back(it->second);
      }
      auto renamed = k::ValueType::get(getContext(), type.getElementType(), type.getShape(),
                                      rewriter.getDenseI64ArrayAttr(axes));
      if (renamed == type) operands.push_back(operand);
      else {
        Value existing;
        for (Operation *user : operand.getUsers()) {
          auto alias = dyn_cast<k::ReshapeOp>(user);
          if (alias && alias.getResult().getType() == renamed &&
              alias.getOrderAttr() == type.getAxisIds() &&
              alias->getBlock() == reshape->getBlock() && alias->isBeforeInBlock(reshape)) {
            existing = alias.getResult();
            break;
          }
        }
        operands.push_back(existing ? existing : k::ReshapeOp::create(rewriter, reshape.getLoc(),
            renamed, operand, type.getAxisIds()).getResult());
      }
    }
    OperationState state(producer->getLoc(), producer->getName());
    state.addOperands(operands);
    state.addTypes(output);
    state.addAttributes(producer->getAttrs());
    rewriter.replaceOp(reshape, rewriter.create(state)->getResults());
    if (producer->use_empty()) rewriter.eraseOp(producer);
    return success();
  }
};

struct EraseUnusedHelperResults : OpRewritePattern<weft::kernel::HelperOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::HelperOp helper,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> retained;
    SmallVector<Type> types;
    for (auto [index, result] : llvm::enumerate(helper.getResults()))
      if (!result.use_empty()) {
        retained.push_back(index);
        types.push_back(result.getType());
      }
    if (retained.size() == helper.getNumResults()) return failure();
    SmallVector<weft::kernel::HelperReturnOp> exits;
    helper.getBody().walk([&](weft::kernel::HelperReturnOp exit) {
      if (exit->getParentOfType<weft::kernel::HelperOp>() == helper) exits.push_back(exit);
    });
    for (auto exit : exits) {
      SmallVector<Value> values;
      for (unsigned index : retained) values.push_back(exit.getValues()[index]);
      rewriter.modifyOpInPlace(exit, [&] { exit.getValuesMutable().assign(values); });
    }
    OperationState state(helper.getLoc(), helper->getName());
    state.addOperands(helper->getOperands());
    state.addTypes(types);
    state.addAttributes(helper->getAttrs());
    state.addRegion();
    Operation *replacement = rewriter.create(state);
    replacement->getRegion(0).takeBody(helper.getBody());
    for (auto [index, original] : llvm::enumerate(retained))
      rewriter.replaceAllUsesWith(helper.getResult(original), replacement->getResult(index));
    rewriter.eraseOp(helper);
    return success();
  }
};

class NormalizeCanonicalControl
    : public PassWrapper<NormalizeCanonicalControl, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeCanonicalControl)
  StringRef getArgument() const override { return "weft-normalize-canonical-control"; }
  StringRef getDescription() const override {
    return "Inline ordinary single-exit helpers and normalize SCF carries before physical lowering";
  }
  void runOnOperation() override {
    SmallVector<weft::kernel::HelperOp> helpers;
    getOperation().walk<WalkOrder::PostOrder>([&](weft::kernel::HelperOp helper) {
      helpers.push_back(helper);
    });
    for (auto helper : helpers) {
      if (helper.getNoinline() || !llvm::hasSingleElement(helper.getBody())) continue;
      Block &body = helper.getBody().front();
      if (body.empty()) continue;
      auto exit = dyn_cast<weft::kernel::HelperReturnOp>(body.back());
      if (!exit) continue;
      bool nestedExit = false;
      helper.getBody().walk([&](weft::kernel::HelperReturnOp ret) {
        if (ret != exit && ret->getParentOfType<weft::kernel::HelperOp>() == helper)
          nestedExit = true;
      });
      if (nestedExit) continue;
      helper.replaceAllUsesWith(exit.getValues());
      exit.erase();
      helper->getBlock()->getOperations().splice(helper->getIterator(), body.getOperations());
      helper.erase();
    }
    SmallVector<scf::ForOp> unrolled;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.unroll")) unrolled.push_back(loop);
    });
    for (auto loop : unrolled) {
      loop->removeAttr("weft.unroll");
      if (failed(loopUnrollFull(loop))) {
        loop.emitError("source unroll requires a statically bounded SCF loop");
        signalPassFailure();
        return;
      }
    }
    SmallVector<Operation *> unrollBounds;
    getOperation().walk([&](Operation *operation) {
      if (isa<arith::AddIOp, arith::MulIOp>(operation)) unrollBounds.push_back(operation);
    });
    GreedyRewriteConfig foldBounds;
    foldBounds.setStrictness(GreedyRewriteStrictness::ExistingOps);
    if (failed(applyOpPatternsGreedily(unrollBounds, RewritePatternSet(&getContext()), foldBounds))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
    scf::WhileOp::getCanonicalizationPatterns(patterns, &getContext());
    patterns.add<PushCoordinateRename, FoldScalarProjection, FoldFullSelect, FoldStaticSubviewOffsets,
                 FoldDescriptorExtent,
                 FoldIdentitySlice, FoldScalarInteger, FoldScalarIntegerCompare,
                 FoldConstantIf, NormalizeScalarSelect, NormalizeBoundedGather,
                 FoldIdentityLookup, FoldIdentityTileGather,
                 EraseUnusedEmpty, EraseUnusedHelperResults>(&getContext());
    // Dead control carries can be the only extra consumer of a coordinate
    // rename. Revisit both in the same bounded worklist, so deleting a carry
    // exposes the producer to the existing single-consumer rewrite. Folding
    // remains disabled here: Canonical domains retain their explicit count
    // relation, including ceildiv(extent, 1), until physical lowering.
    GreedyRewriteConfig configuration;
    configuration.enableFolding(false).enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), configuration))) {
      signalPassFailure();
      return;
    }
    IRRewriter rewriter(&getContext());
    // CSE may share pure numerical expressions, but two source declarations
    // still represent two births. Keep their identities distinct during the
    // generic transform, without adding an attribute to the resulting IR.
    int64_t birth = 0;
    getOperation().walk([&](Operation *op) {
      if (isa<weft::kernel::NewOp, weft::kernel::MaterializeOp>(op))
        op->setAttr("weft.cse_birth", rewriter.getI64IntegerAttr(birth++));
    });
    DominanceInfo dominance(getOperation());
    eliminateCommonSubExpressions(rewriter, dominance, getOperation());
    getOperation().walk([](Operation *op) { op->removeAttr("weft.cse_birth"); });
  }
};
}

std::unique_ptr<Pass> weft::createNormalizeCanonicalControlPass() {
  return std::make_unique<NormalizeCanonicalControl>();
}
