#include "Weft/Analysis/LogicalAxes.h"
#include "Weft/Target/RISCVPasses.h"
#include "RISCVPhysicalSupport.h"
#include "NumericalValidity.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
// Preserve the closed grouped/widen/reduce relation consumed by the existing
// widening-dot materializer. All other consumers need the grouped values.
bool hasWideningReduction(riscv::MacGroupsOp op) {
  auto a = dyn_cast<IntegerType>(logicalElement(op.getLhs().getType()));
  auto c = dyn_cast<IntegerType>(logicalElement(op.getRhs().getType()));
  if (!a || !c || a.getWidth() > 8 || c.getWidth() > 8 ||
      a.isUnsigned() == c.isUnsigned() || !logicalElement(op.getResult().getType()).isInteger(16) ||
      !op.getResult().hasOneUse()) return false;
  auto widen = dyn_cast<riscv::WidenOp>(*op.getResult().getUsers().begin());
  if (!widen || !logicalElement(widen.getResult().getType()).isInteger(32) ||
      !widen.getResult().hasOneUse()) return false;
  auto reduce = dyn_cast<riscv::ReduceOp>(*widen.getResult().getUsers().begin());
  return reduce && reduce.getKind() == "add";
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
    SmallVector<riscv::MacGroupsOp> operations;
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
      if (auto mac = dyn_cast<riscv::MacGroupsOp>(op); mac && !hasWideningReduction(mac)) operations.push_back(mac);
    });
    IRRewriter b(&getContext());
    DenseMap<Value, Value> results, masks;
    for (auto op : operations)
      if (failed(materialize(b, op, nextAxis++, results, masks))) { signalPassFailure(); return; }
    if (failed(propagateNumericValidity(b, original, masks, nextAxis))) { signalPassFailure(); return; }
    for (auto op : operations) b.replaceOp(op, results.lookup(op.getResult()));
  }
private:
  LogicalResult materialize(IRRewriter &b, riscv::MacGroupsOp op, int64_t groupedAxis,
                            DenseMap<Value, Value> &results, DenseMap<Value, Value> &masks) {
    auto left = cast<riscv::ValueType>(op.getLhs().getType());
    auto right = cast<riscv::ValueType>(op.getRhs().getType());
    auto result = cast<riscv::ValueType>(op.getResult().getType());
    auto domain = [](riscv::ValueType type) {
      return analysis::LogicalDomain{type.getShape().asArrayRef(), type.getAxisIds().asArrayRef()};
    };
    auto relation = analysis::groupedAxes({domain(left), domain(right)}, domain(result), op.getGroup());
    if (!relation) return op.emitError("grouped arithmetic requires a bounded complete group and matching logical axes");
    if (llvm::any_of(left.getShape().asArrayRef(), [](int64_t n) { return n > UINT32_MAX; }) ||
        llvm::any_of(right.getShape().asArrayRef(), [](int64_t n) { return n > UINT32_MAX; }))
      return op.emitError("grouped projection exceeds its u32 coordinate domain");
    b.setInsertionPoint(op);
    Location loc = op.getLoc();
    SmallVector<SmallVector<Value>> bounds;
    for (Value input : {op.getLhs(), op.getRhs()}) {
      auto active = activeNumericExtents(b, input);
      if (failed(active)) return op.emitError("grouped arithmetic requires a complete active-coordinate relation for each input");
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
    for (auto type : {left, right})
      for (int64_t sourceAxis : type.getAxisIds().asArrayRef()) {
        int64_t axis = sourceAxis == relation->axis ? groupedAxis : sourceAxis;
        if (!llvm::is_contained(productAxes, axis)) {
          productAxes.push_back(axis);
          productShape.push_back(result.getShape()[llvm::find(outputAxes, axis) - outputAxes.begin()]);
        }
      }
    auto accumulatorType = makeType(result.getElementType(), productShape, productAxes);
    auto predicateType = makeType(b.getI1Type(), productShape, productAxes);
    auto product = [&](Value member) -> std::pair<Value, Value> {
      Value selected = riscv::BinaryOp::create(b, loc, indexType, bases, member, "add", unselectedLeaf(b));
      SmallVector<Value> values;
      Value valid = riscv::ConstantOp::create(b, loc, b.getI1Type(), b.getBoolAttr(true));
      SmallVector<int64_t> validityShape, validityAxes;
      for (auto [inputNumber, input] : llvm::enumerate(SmallVector<Value>{op.getLhs(), op.getRhs()})) {
        auto type = cast<riscv::ValueType>(input.getType());
        SmallVector<int64_t> shape(type.getShape().asArrayRef()), axes(type.getAxisIds().asArrayRef());
        shape.back() = relation->groups; axes.back() = groupedAxis;
        for (auto [dimension, axis] : llvm::enumerate(axes)) {
          Value coordinates = dimension + 1 == axes.size() ? selected :
              Value(riscv::IotaOp::create(b, loc, makeType(u32, {shape[dimension]}, {axis}),
                                        0, shape[dimension], unselectedLeaf(b)));
          Value active = riscv::CastOp::create(b, loc, u32, bounds[inputNumber][dimension], unselectedLeaf(b));
          Value predicate = riscv::CompareOp::create(b, loc,
              makeType(b.getI1Type(), {shape[dimension]}, {axis}), coordinates, active, "lt", unselectedLeaf(b));
          auto position = llvm::find(validityAxes, axis);
          if (position == validityAxes.end()) {
            validityAxes.push_back(axis); validityShape.push_back(shape[dimension]);
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
      Value value = riscv::BinaryOp::create(b, loc, accumulatorType, values[0], values[1], "mul", unselectedLeaf(b));
      Attribute zero = isa<FloatType>(result.getElementType()) ? Attribute(b.getFloatAttr(result.getElementType(), 0.0)) :
          Attribute(b.getIntegerAttr(result.getElementType(), 0));
      Value identity = riscv::ConstantOp::create(b, loc, result.getElementType(), zero);
      value = riscv::SelectOp::create(b, loc, accumulatorType, valid, value, identity, unselectedLeaf(b));
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
      auto loop = scf::ForOp::create(b, loc, lower, upper, lower, ValueRange{accumulator, active});
      copyOrigin(op, loop);
      {
        OpBuilder::InsertionGuard guard(b);
        if (!loop.getBody()->empty()) b.eraseOp(loop.getBody()->getTerminator());
        b.setInsertionPointToStart(loop.getBody());
        Value member = riscv::CastOp::create(b, loc, u32, loop.getInductionVar(), unselectedLeaf(b));
        auto [term, valid] = product(member);
        Value value = riscv::BinaryOp::create(b, loc, accumulatorType, loop.getRegionIterArg(0),
            term, "add", unselectedLeaf(b));
        Value any = riscv::BinaryOp::create(b, loc, predicateType, loop.getRegionIterArg(1), valid, "or", unselectedLeaf(b));
        scf::YieldOp::create(b, loc, ValueRange{value, any});
      }
      accumulator = loop.getResult(0);
      active = loop.getResult(1);
    }
    auto orderedType = makeType(result.getElementType(), result.getShape().asArrayRef(), outputAxes);
    if (productAxes != outputAxes)
      accumulator = riscv::ReshapeOp::create(b, loc, orderedType, accumulator,
          b.getDenseI64ArrayAttr(outputAxes), DenseI64ArrayAttr());
    auto orderedMask = makeType(b.getI1Type(), result.getShape().asArrayRef(), outputAxes);
    if (productAxes != outputAxes)
      active = riscv::ReshapeOp::create(b, loc, orderedMask, active,
          b.getDenseI64ArrayAttr(outputAxes), DenseI64ArrayAttr());
    auto restored = riscv::ReshapeOp::create(b, loc, result, accumulator,
        orderedType.getAxisIds(), DenseI64ArrayAttr());
    copyOrigin(op, restored);
    results[op.getResult()] = restored;
    masks[op.getResult()] = riscv::ReshapeOp::create(b, loc,
        makeType(b.getI1Type(), result.getShape().asArrayRef(), result.getAxisIds().asArrayRef()), active,
        orderedMask.getAxisIds(), DenseI64ArrayAttr());
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> weft::createMaterializeRISCVGroupedArithmeticPass() {
  return std::make_unique<MaterializeRISCVGroupedArithmeticPass>();
}
