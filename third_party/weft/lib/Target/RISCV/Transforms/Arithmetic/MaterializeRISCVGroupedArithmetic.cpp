#include "Weft/Analysis/LogicalAxes.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Target/RISCV/RISCVPasses.h"
#include "Transforms/Support/RISCVPhysicalSupport.h"
#include "Transforms/Representation/NumericalValidity.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
// Memory planning expands noncontiguous encoded snapshots into numerical
// fields. Materialize their grouped operations before assigning layouts, while
// both sides of the new projections still have one consistent layout state.
bool needsNumericalSnapshot(Value value) {
  auto field = sourceField(value);
  if (!field) return false;
  value = field.getOwner();
  for (unsigned depth = 0; depth < 32; ++depth) {
    if (auto load = value.getDefiningOp<riscv::LoadOp>())
      return !riscv::isContiguousRecordRegion(load.getRegion()) &&
             needsReadSnapshot(load);
    if (auto extract = value.getDefiningOp<riscv::ExtractOp>()) value = extract.getInput();
    else if (auto stage = value.getDefiningOp<riscv::MaterializeOp>()) {
      // An encoded stage owns a saved record, including its field geometry.
      // Later writes to the original view do not require a numerical snapshot
      // of that private record's consumers.
      if (isa<kernel::EncodingType>(logicalElement(stage.getResult().getType())))
        return false;
      value = stage.getInput();
    }
    else if (auto stage = value.getDefiningOp<riscv::RegisterMaterializeOp>()) value = stage.getInput();
    else return false;
  }
  return true;
}
// Preserve the closed grouped/widen/reduce relation consumed by the existing
// widening-dot materializer. All other consumers need the grouped values.
bool hasWideningReduction(riscv::MacGroupsOp op) {
  if (needsNumericalSnapshot(op.getLhs()) ||
      needsNumericalSnapshot(op.getRhs())) return false;
  auto a = dyn_cast<IntegerType>(logicalElement(op.getLhs().getType()));
  auto c = dyn_cast<IntegerType>(logicalElement(op.getRhs().getType()));
  if (!a || !c || a.getWidth() > 8 || c.getWidth() > 8 ||
      a.isUnsigned() == c.isUnsigned() || !logicalElement(op.getResult().getType()).isInteger(16) ||
      !op.getResult().hasOneUse()) return false;
  // The fused leaf consumes a layered encoded field. A staged numerical
  // value has already severed that storage relation and must use the ordinary
  // grouped projection below, retaining the same finite-width boundary.
  auto field = sourceField(a.isUnsigned() ? op.getLhs() : op.getRhs());
  auto facts = field ? fieldFacts(field) : FieldFacts{};
  if (!field || facts.mapping == "dense") return false;
  auto widen = dyn_cast<riscv::WidenOp>(*op.getResult().getUsers().begin());
  if (!widen || !logicalElement(widen.getResult().getType()).isInteger(32) ||
      !widen.getResult().hasOneUse()) return false;
  auto reduce = dyn_cast<riscv::ReduceOp>(*widen.getResult().getUsers().begin());
  if (!reduce || reduce.getKind() != "add") return false;
  auto result = dyn_cast<riscv::ValueType>(reduce.getResult().getType());
  // The cohort leaf vectorizes a free output axis. Singleton outputs instead
  // use the general grouped values followed by the ordinary lane reduction.
  return result && llvm::any_of(result.getShape().asArrayRef(),
                               [](int64_t extent) { return extent > 1; });
}

class MaterializeRISCVGroupedArithmeticPass
    : public PassWrapper<MaterializeRISCVGroupedArithmeticPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVGroupedArithmeticPass)
  StringRef getArgument() const override { return "weft-riscv-materialize-grouped-arithmetic"; }
  StringRef getDescription() const override { return "Materialize closed grouped arithmetic with explicit axis projections and numeric boundaries"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<riscv::WEFTRISCVDialect, arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    versionSharedGroupedPrograms();
    foldExactGroupedReductions();
    SmallVector<Operation *> operations;
    SmallVector<Operation *> original;
    int64_t nextAxis = 0;
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      original.push_back(op);
      auto inspect = [&](Type type) {
        for (int64_t axis : logicalAxes(type)) nextAxis = std::max(nextAxis, axis + 1);
        if (auto point = dyn_cast<riscv::PointType>(type))
          nextAxis = std::max(nextAxis, point.getDomain().getAxisId() + 1);
        if (auto domain = dyn_cast<riscv::DomainType>(type))
          nextAxis = std::max(nextAxis, domain.getAxisId() + 1);
      };
      for (Type type : op->getOperandTypes()) inspect(type);
      for (Type type : op->getResultTypes()) inspect(type);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Type type : block.getArgumentTypes()) inspect(type);
      if (auto mac = dyn_cast<riscv::MacGroupsOp>(op);
          mac && !hasWideningReduction(mac))
        operations.push_back(mac);
      if (auto fold = dyn_cast<riscv::Fold2Op>(op)) {
        auto field = sourceField(fold.getInput());
        auto facts = field ? fieldFacts(field) : FieldFacts{};
        // A numerical snapshot is no longer an encoded field address. Keep the
        // direct natural-field leaf only when its original storage edge exists;
        // otherwise materialize the same pair sums through numeric projections.
        if (needsNumericalSnapshot(fold.getInput()) || !field ||
            facts.mapping != "natural" || facts.bitOffset % 8 ||
            !logicalElement(fold.getInput().getType()).isSignedInteger(16))
          operations.push_back(fold);
      }
    });
    IRRewriter b(&getContext());
    DenseMap<Value, Value> results, masks;
    for (auto op : operations) {
      int64_t groupedAxis = nextAxis++;
      if (failed(materialize(b, op, groupedAxis, nextAxis, results, masks))) { signalPassFailure(); return; }
    }
    if (failed(propagateNumericValidity(b, original, masks, nextAxis))) { signalPassFailure(); return; }
    for (auto op : operations) b.replaceOp(op, results.lookup(op->getResult(0)));
  }
private:
  void versionSharedGroupedPrograms() {
    SmallVector<riscv::MacGroupsOp> candidates;
    getOperation().walk([&](riscv::MacGroupsOp mac) {
      if (hasWideningReduction(mac)) candidates.push_back(mac);
    });
    DenseSet<Operation *> handled;
    IRRewriter b(&getContext());
    for (auto mac : candidates) {
      auto unsignedType = dyn_cast<IntegerType>(logicalElement(mac.getLhs().getType()));
      auto field = sourceField(unsignedType && unsignedType.isUnsigned() ? mac.getLhs() : mac.getRhs());
      Value owner = field ? field.getOwner() : Value();
      riscv::MaterializeOp stage;
      for (unsigned depth = 0; owner && depth < 32; ++depth) {
        if ((stage = owner.getDefiningOp<riscv::MaterializeOp>())) break;
        if (auto extract = owner.getDefiningOp<riscv::ExtractOp>()) owner = extract.getInput();
        else break;
      }
      if (!stage || handled.contains(stage)) continue;
      auto saved = dyn_cast<riscv::ValueType>(stage.getResult().getType());
      auto encoding = saved ? dyn_cast<kernel::EncodingType>(saved.getElementType()) : kernel::EncodingType();
      auto declaration = encoding ? findEncoding(stage, encoding.getFamily()) : riscv::EncodingDeclOp();
      if (!saved || saved.getShape().size() != 2 || !declaration ||
          stage->getBlock()->getTerminator()->getNumOperands()) continue;
      riscv::LoopOp reuseLoop;
      for (Operation *parent = mac->getParentOp(); parent && parent != stage->getParentOp(); parent = parent->getParentOp())
        if (parent->getBlock() == stage->getBlock()) reuseLoop = dyn_cast<riscv::LoopOp>(parent);
      auto domain = reuseLoop ? reuseLoop.getDomain().getDefiningOp<riscv::DomainOp>() : riscv::DomainOp();
      if (!domain || llvm::is_contained(saved.getAxisIds().asArrayRef(),
                                       domain.getResult().getType().getAxisId())) continue;
      Value extent = domain.getExtent();
      if (auto *definition = extent.getDefiningOp(); definition &&
          definition->getBlock() == stage->getBlock() && !definition->isBeforeInBlock(stage)) continue;
      auto partial = cast<riscv::ValueType>(mac.getResult().getType());
      int64_t outputCount = 1, frontier = 0;
      for (size_t i = 0; i + 1 < partial.getShape().size(); ++i) {
        outputCount *= partial.getShape()[i];
        if (partial.getAxisIds()[i] == saved.getAxisIds()[0]) frontier = partial.getShape()[i];
      }
      const int64_t rows = saved.getShape()[0], elements = saved.getShape()[1];
      const int64_t terms = partial.getShape().asArrayRef().back() * mac.getGroup();
      const int64_t recordElements = declaration.getElements();
      const int64_t recordBytes = declaration.getStorageBits() / 8;
      if (outputCount <= 0 || frontier <= 1 || rows <= 0 || elements <= 0 ||
          recordElements <= 0 || recordBytes <= 0 || terms <= 0 || elements % recordElements) continue;
      auto target = mac->getParentOfType<riscv::KernelOp>().getTarget();
      const int64_t byteLanes = std::max<int64_t>(1, target.getVlenBits() / 8);
      const int64_t partialLanes = std::max<int64_t>(1, target.getVlenBits() / 16);
      auto ceilDiv = [](int64_t x, int64_t y) { return x / y + (x % y != 0); };
      const int64_t cohorts = ceilDiv(rows, frontier);
      const int64_t records = rows * (elements / recordElements);
      const int64_t encodedPreparation = recordBytes * (elements / recordElements) * cohorts * (frontier + 2);
      const int64_t decodedPreparation = records * ceilDiv(recordBytes, byteLanes) * 2;
      const int64_t repeats = cohorts * ceilDiv(elements, terms);
      const int64_t encodedWork = ceilDiv(outputCount, partialLanes) *
          ceilDiv(terms, mac.getGroup()) * (mac.getGroup() + 2) * repeats;
      const int64_t decodedWork = outputCount * ceilDiv(terms, byteLanes) * 5 * repeats;
      if (encodedPreparation <= decodedPreparation || decodedWork <= encodedWork) continue;
      const int64_t threshold = (encodedPreparation - decodedPreparation - 1) / (decodedWork - encodedWork);
      if (threshold <= 0) continue;
      auto scope = stage->getParentOfType<riscv::LoopOp>();
      if (!scope || scope.getNumResults() ||
          (extent.getDefiningOp() && scope->isProperAncestor(extent.getDefiningOp()))) continue;
      handled.insert(stage);
      b.setInsertionPoint(scope);
      auto limit = arith::ConstantIndexOp::create(b, stage.getLoc(), threshold);
      auto condition = arith::CmpIOp::create(b, stage.getLoc(), arith::CmpIPredicate::ule, extent, limit);
      auto choice = scf::IfOp::create(b, stage.getLoc(), condition, true);
      choice->setAttr("weft.riscv.grouped_reuse_choice", b.getDenseI64ArrayAttr(
          {threshold, encodedPreparation, decodedPreparation, encodedWork, decodedWork}));
      auto clone = [&](Block &block, StringRef strategy) {
        IRMapping mapping;
        b.setInsertionPoint(block.getTerminator());
        Operation *copy = b.clone(*scope.getOperation(), mapping);
        copy->walk([&](riscv::MacGroupsOp grouped) {
          grouped->setAttr("weft.riscv.grouped_strategy", b.getStringAttr(strategy));
        });
      };
      clone(choice.getThenRegion().front(), "decoded");
      clone(choice.getElseRegion().front(), "encoded");
      scope.erase();
      break; // One shared stage per invocation; other closed groups still lower normally.
    }
  }

  void foldExactGroupedReductions() {
    SmallVector<riscv::MacGroupsOp> operations;
    getOperation().walk([&](riscv::MacGroupsOp op) { operations.push_back(op); });
    IRRewriter b(&getContext());
    for (auto mac : operations) {
      if (!mac.getResult().hasOneUse()) continue;
      auto partial = dyn_cast<riscv::ValueType>(mac.getResult().getType());
      auto width = partial ? dyn_cast<IntegerType>(partial.getElementType()) : IntegerType();
      auto widen = dyn_cast<riscv::WidenOp>(*mac.getResult().getUsers().begin());
      if (!width || !width.isSigned() || width.getWidth() != 16 || !widen ||
          logicalBitWidth(mac.getLhs().getType()) > 16 ||
          logicalBitWidth(mac.getRhs().getType()) > 16 ||
          !widen.getResult().hasOneUse() || !logicalElement(widen.getResult().getType()).isSignedInteger(32))
        continue;
      auto reduce = dyn_cast<riscv::ReduceOp>(*widen.getResult().getUsers().begin());
      if (!reduce || reduce.getKind() != "add" ||
          reduce.getAxis() != int64_t(partial.getShape().size()) - 1) continue;
      auto lhs = integerRange(mac.getLhs()), rhs = integerRange(mac.getRhs());
      if (!lhs || !rhs) continue;
      const int64_t amin = lhs->minimum.getExtValue(), amax = lhs->maximum.getExtValue();
      const int64_t bmin = rhs->minimum.getExtValue(), bmax = rhs->maximum.getExtValue();
      __int128 products[] = {__int128(amin) * bmin, __int128(amin) * bmax,
                            __int128(amax) * bmin, __int128(amax) * bmax};
      auto bounds = std::minmax_element(std::begin(products), std::end(products));
      __int128 minimum = *bounds.first * mac.getGroup();
      __int128 maximum = *bounds.second * mac.getGroup();
      if (minimum < -32768 || maximum > 32767) continue;
      if (hasWideningReduction(mac)) {
        auto kernel = mac->getParentOfType<riscv::KernelOp>();
        auto outputs = cast<riscv::ValueType>(reduce.getResult().getType());
        int64_t outputCount = 1;
        for (int64_t extent : outputs.getShape().asArrayRef()) {
          if (extent <= 0 || outputCount > INT32_MAX / extent) {
            outputCount = INT32_MAX;
            break;
          }
          outputCount *= extent;
        }
        const int64_t terms = partial.getShape().asArrayRef().back() * mac.getGroup();
        const int64_t byteLanes = std::max<int64_t>(1, kernel.getTarget().getVlenBits() / 8);
        const int64_t partialLanes = std::max<int64_t>(1, kernel.getTarget().getVlenBits() / 16);
        auto ceilDiv = [](int64_t value, int64_t divisor) {
          return value / divisor + (value % divisor != 0);
        };
        // Estimate encoded cohort updates against per-record decode, widening
        // multiply and reduction. Costs depend on axes and target vector
        // capacity, never on an encoding name or a benchmark configuration.
        const int64_t encodedIssues = ceilDiv(outputCount, partialLanes) *
            ceilDiv(terms, mac.getGroup()) * (mac.getGroup() + 2);
        const int64_t decodedIssues = outputCount * ceilDiv(terms, byteLanes) * 5;
        auto strategy = mac->getAttrOfType<StringAttr>("weft.riscv.grouped_strategy");
        const bool forcedEncoded = strategy && strategy.getValue() == "encoded";
        const bool forcedDecoded = strategy && strategy.getValue() == "decoded";
        const bool keepEncoded = forcedEncoded || (!forcedDecoded && encodedIssues <= decodedIssues);
        mac->setAttr("weft.riscv.grouped_candidate_cost", b.getDenseI64ArrayAttr(
            {outputCount, terms, encodedIssues, decodedIssues}));
        if (keepEncoded) continue;
      }
      b.setInsertionPoint(reduce);
      auto dot = riscv::DotOp::create(b, reduce.getLoc(), reduce.getResult().getType(),
          mac.getLhs(), mac.getRhs(), b.getDenseI64ArrayAttr({partial.getAxisIds().asArrayRef().back()}),
          TypeAttr::get(logicalElement(widen.getResult().getType())),
          "unassigned", "unassigned", mac.getSchedule(), unselectedLeaf(b));
      // The boundary is eliminated only after proving that every narrow group
      // has the same signed value before and after widening. The outer i32 sum
      // then has exactly the same modular result, including partial tails.
      dot->setAttr("engine", b.getStringAttr("vector"));
      dot->setAttr("weft.riscv.proven_group_range", b.getDenseI64ArrayAttr(
          {int64_t(mac.getGroup()), int64_t(minimum), int64_t(maximum)}));
      copyOrigin(mac, dot);
      b.replaceOp(reduce, dot.getResult());
      b.eraseOp(widen);
      b.eraseOp(mac);
    }
  }

  LogicalResult materialize(IRRewriter &b, Operation *op, int64_t groupedAxis, int64_t &nextAxis,
                            DenseMap<Value, Value> &results, DenseMap<Value, Value> &masks) {
    SmallVector<Value> inputs(op->getOperands());
    auto mac = dyn_cast<riscv::MacGroupsOp>(op);
    int64_t group = mac ? mac.getGroup() : 2;
    auto result = cast<riscv::ValueType>(op->getResult(0).getType());
    auto domain = [](riscv::ValueType type) {
      return analysis::LogicalDomain{type.getShape().asArrayRef(), type.getAxisIds().asArrayRef()};
    };
    SmallVector<analysis::LogicalDomain> domains;
    for (Value input : inputs) domains.push_back(domain(cast<riscv::ValueType>(input.getType())));
    auto relation = analysis::groupedAxes(domains, domain(result), group);
    if (!relation) return op->emitError("grouped arithmetic requires a bounded complete group and matching logical axes");
    for (const auto &input : domains)
      if (llvm::any_of(input.shape, [](int64_t n) { return n > UINT32_MAX; }))
        return op->emitError("grouped projection exceeds its u32 coordinate domain");
    b.setInsertionPoint(op);
    Location loc = op->getLoc();
    SmallVector<SmallVector<Value>> bounds;
    bool complete = true;
    for (Value input : inputs) {
      auto active = activeNumericExtents(b, input);
      if (failed(active)) return op->emitError("grouped arithmetic requires a complete active-coordinate relation for each input");
      auto type = cast<riscv::ValueType>(input.getType());
      for (auto [bound, extent] : llvm::zip(*active, type.getShape().asArrayRef()))
        complete &= extent > 0 && riscv::denseConstantIndex(bound) == std::optional<int64_t>(extent);
      bounds.push_back(*active);
    }
    auto makeType = [&](Type element, ArrayRef<int64_t> shape, ArrayRef<int64_t> axes) {
      return riscv::ValueType::get(b.getContext(), element, b.getDenseI64ArrayAttr(shape),
          b.getDenseI64ArrayAttr(axes), unassignedLayout(b, axes));
    };
    auto u32 = IntegerType::get(b.getContext(), 32, IntegerType::Unsigned);
    auto indexType = makeType(u32, {relation->groups}, {groupedAxis});
    Value index = riscv::IotaOp::create(b, loc, indexType, 0, relation->groups, unselectedLeaf(b));
    Value factor = riscv::ConstantOp::create(b, loc, u32, b.getIntegerAttr(u32, relation->group));
    Value bases = riscv::BinaryOp::create(b, loc, indexType, index, factor, "mul", unselectedLeaf(b));
    SmallVector<int64_t> outputAxes(result.getAxisIds().asArrayRef());
    outputAxes.back() = groupedAxis;
    SmallVector<int64_t> productAxes, productShape;
    for (Value input : inputs) {
      auto type = cast<riscv::ValueType>(input.getType());
      for (int64_t sourceAxis : type.getAxisIds().asArrayRef()) {
        int64_t axis = sourceAxis == relation->axis ? groupedAxis : sourceAxis;
        if (!llvm::is_contained(productAxes, axis)) {
          productAxes.push_back(axis);
          productShape.push_back(result.getShape()[llvm::find(outputAxes, axis) - outputAxes.begin()]);
        }
      }
    }
    auto accumulatorType = makeType(result.getElementType(), productShape, productAxes);
    // Every validity coordinate must be evaluated, including false rows on a
    // free-axis tail. Which free axis becomes an RVV lane is a later decision.
    DenseMap<int64_t, int64_t> predicateAxisMap;
    SmallVector<int64_t> predicateAxes;
    for (int64_t axis : productAxes) {
      int64_t complete = axis == groupedAxis ? groupedAxis : nextAxis++;
      predicateAxisMap[axis] = complete;
      predicateAxes.push_back(complete);
    }
    auto predicateType = makeType(b.getI1Type(), productShape, predicateAxes);
    auto product = [&](Value member) -> std::pair<Value, Value> {
      Value selected = riscv::BinaryOp::create(b, loc, indexType, bases, member, "add", unselectedLeaf(b));
      SmallVector<Value> values;
      Value valid = complete ? Value() : riscv::ConstantOp::create(b, loc, b.getI1Type(), b.getBoolAttr(true));
      SmallVector<int64_t> validityShape, validityAxes;
      for (auto [inputNumber, input] : llvm::enumerate(inputs)) {
        auto type = cast<riscv::ValueType>(input.getType());
        SmallVector<int64_t> shape(type.getShape().asArrayRef()), axes(type.getAxisIds().asArrayRef());
        shape.back() = relation->groups; axes.back() = groupedAxis;
        for (auto [dimension, axis] : llvm::enumerate(axes)) {
          if (complete) continue;
          int64_t completeAxis = predicateAxisMap.lookup(axis);
          Value coordinates = dimension + 1 == axes.size() ? selected :
              Value(riscv::IotaOp::create(b, loc, makeType(u32, {shape[dimension]}, {completeAxis}),
                                        0, shape[dimension], unselectedLeaf(b)));
          Value active = riscv::CastOp::create(b, loc, u32, bounds[inputNumber][dimension], unselectedLeaf(b));
          Value predicate = riscv::CompareOp::create(b, loc,
              makeType(b.getI1Type(), {shape[dimension]}, {completeAxis}), coordinates, active, "lt", unselectedLeaf(b));
          auto position = llvm::find(validityAxes, completeAxis);
          if (position == validityAxes.end()) {
            validityAxes.push_back(completeAxis); validityShape.push_back(shape[dimension]);
          } else {
            auto &extent = validityShape[position - validityAxes.begin()];
            extent = std::max(extent, shape[dimension]);
          }
          valid = riscv::BinaryOp::create(b, loc, makeType(b.getI1Type(), validityShape, validityAxes),
                                        valid, predicate, "and", unselectedLeaf(b));
        }
        SmallVector<Attribute> selectors(shape.size(), b.getStringAttr("all"));
        selectors.back() = b.getStringAttr("gather");
        Value projected = riscv::ExtractOp::create(b, loc, makeType(type.getElementType(), shape, axes),
            input, ValueRange{selected}, b.getArrayAttr(selectors), unassignedAccess(b), unselectedLeaf(b));
        if (type.getElementType() != result.getElementType())
          projected = riscv::CastOp::create(b, loc, makeType(result.getElementType(), shape, axes),
              projected, unselectedLeaf(b));
        values.push_back(projected);
      }
      Value value = mac ? Value(riscv::BinaryOp::create(b, loc, accumulatorType, values[0], values[1], "mul", unselectedLeaf(b))) : values[0];
      if (complete) return {value, Value()};
      Attribute zero = isa<FloatType>(result.getElementType()) ? Attribute(b.getFloatAttr(result.getElementType(), 0.0)) :
          Attribute(b.getIntegerAttr(result.getElementType(), 0));
      Value identity = riscv::ConstantOp::create(b, loc, result.getElementType(), zero);
      Value numericMask = riscv::ReshapeOp::create(b, loc, makeType(b.getI1Type(), productShape, productAxes),
          valid, predicateType.getAxisIds(), DenseI64ArrayAttr());
      value = riscv::SelectOp::create(b, loc, accumulatorType, numericMask, value, identity, unselectedLeaf(b));
      return {value, valid};
    };
    auto [first, active] = product(riscv::ConstantOp::create(b, loc, u32, b.getIntegerAttr(u32, 0)));
    Attribute zero = isa<FloatType>(result.getElementType())
        ? Attribute(b.getFloatAttr(result.getElementType(), 0.0)) : Attribute(b.getIntegerAttr(result.getElementType(), 0));
    Value initial = riscv::ConstantOp::create(b, loc, result.getElementType(), zero);
    Value accumulator = riscv::BinaryOp::create(b, loc, accumulatorType, initial, first, "add", unselectedLeaf(b));
    if (relation->group > 1) {
      auto lower = arith::ConstantIndexOp::create(b, loc, 1);
      auto upper = arith::ConstantIndexOp::create(b, loc, relation->group);
      SmallVector<Value> carried{accumulator};
      if (!complete) carried.push_back(active);
      auto loop = scf::ForOp::create(b, loc, lower, upper, lower, carried);
      copyOrigin(op, loop);
      {
        OpBuilder::InsertionGuard guard(b);
        if (!loop.getBody()->empty()) b.eraseOp(loop.getBody()->getTerminator());
        b.setInsertionPointToStart(loop.getBody());
        Value member = riscv::CastOp::create(b, loc, u32, loop.getInductionVar(), unselectedLeaf(b));
        auto [term, valid] = product(member);
        Value value = riscv::BinaryOp::create(b, loc, accumulatorType, loop.getRegionIterArg(0),
            term, "add", unselectedLeaf(b));
        SmallVector<Value> yielded{value};
        if (!complete)
          yielded.push_back(riscv::BinaryOp::create(b, loc, predicateType,
              loop.getRegionIterArg(1), valid, "or", unselectedLeaf(b)));
        scf::YieldOp::create(b, loc, yielded);
      }
      accumulator = loop.getResult(0);
      if (!complete) active = loop.getResult(1);
    }
    auto orderedType = makeType(result.getElementType(), result.getShape().asArrayRef(), outputAxes);
    if (productAxes != outputAxes)
      accumulator = riscv::ReshapeOp::create(b, loc, orderedType, accumulator,
          b.getDenseI64ArrayAttr(outputAxes), DenseI64ArrayAttr());
    SmallVector<int64_t> orderedMaskAxes;
    for (int64_t axis : outputAxes) orderedMaskAxes.push_back(predicateAxisMap.lookup(axis));
    auto orderedMask = makeType(b.getI1Type(), result.getShape().asArrayRef(), orderedMaskAxes);
    if (!complete && productAxes != outputAxes)
      active = riscv::ReshapeOp::create(b, loc, orderedMask, active,
          b.getDenseI64ArrayAttr(orderedMaskAxes), DenseI64ArrayAttr());
    auto restored = riscv::ReshapeOp::create(b, loc, result, accumulator,
        orderedType.getAxisIds(), DenseI64ArrayAttr());
    copyOrigin(op, restored);
    results[op->getResult(0)] = restored;
    if (!complete)
      masks[op->getResult(0)] = riscv::ReshapeOp::create(b, loc,
          makeType(b.getI1Type(), result.getShape().asArrayRef(), result.getAxisIds().asArrayRef()), active,
          orderedMask.getAxisIds(), DenseI64ArrayAttr());
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> weft::createMaterializeRISCVGroupedArithmeticPass() {
  return std::make_unique<MaterializeRISCVGroupedArithmeticPass>();
}
