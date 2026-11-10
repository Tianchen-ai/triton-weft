#include "Transforms/Support/RISCVPhysicalSupport.h"
#include "Weft/Target/RISCV/RISCVPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
// The source combiner owns the numerical operation. This pass only chooses a
// tree inside that closed primitive; adjacent pairs preserve operand order.
class MaterializeRISCVReductionsPass
    : public PassWrapper<MaterializeRISCVReductionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVReductionsPass)
  StringRef getArgument() const override { return "weft-riscv-materialize-reductions"; }
  StringRef getDescription() const override { return "Materialize ordered tuple-combiner reduction trees"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<riscv::WEFTRISCVDialect, arith::ArithDialect>();
  }
  void runOnOperation() override {
    materializeIntegerReductionContracts();
    SmallVector<riscv::ReduceRegionOp> reductions;
    SmallVector<riscv::ScanOp> scans;
    int64_t nextAxis = 0;
    auto inspect = [&](Type type) {
      if (auto value = dyn_cast<riscv::ValueType>(type))
        for (int64_t axis : value.getAxisIds().asArrayRef()) nextAxis = std::max(nextAxis, axis + 1);
    };
    getOperation().walk([&](Operation *op) {
      for (Type type : op->getOperandTypes()) inspect(type);
      for (Type type : op->getResultTypes()) inspect(type);
      if (auto reduce = dyn_cast<riscv::ReduceRegionOp>(op)) reductions.push_back(reduce);
      if (auto scan = dyn_cast<riscv::ScanOp>(op)) scans.push_back(scan);
    });
    IRRewriter builder(&getContext());
    for (auto reduce : reductions)
      if (failed(materialize(builder, reduce, nextAxis))) { signalPassFailure(); return; }
    for (auto scan : scans)
      if (failed(materializeScan(builder, scan, nextAxis))) { signalPassFailure(); return; }
  }

private:
  void materializeIntegerReductionContracts() {
    SmallVector<Operation *> contracts;
    getOperation().walk([&](Operation *op) {
      if (!isa<riscv::DotOp, riscv::ContractOp, riscv::OuterContractOp>(op)) return;
      auto intent = op->getAttrOfType<StringAttr>("engine");
      if (intent && intent.getValue() == "matrix") return;
      auto result = dyn_cast<riscv::ValueType>(op->getResult(0).getType());
      auto element = dyn_cast<IntegerType>(logicalElement(op->getResult(0).getType()));
      auto lhs = dyn_cast<riscv::ValueType>(op->getOperand(0).getType());
      auto rhs = dyn_cast<riscv::ValueType>(op->getOperand(1).getType());
      auto over = op->getAttrOfType<DenseI64ArrayAttr>("over");
      if (!element || !lhs || !rhs || !over || over.size() != 1 ||
          (result && llvm::any_of(result.getShape().asArrayRef(),
                                 [](int64_t n) { return n != 1; })))
        return;
      auto leftElement = dyn_cast<IntegerType>(lhs.getElementType());
      auto rightElement = dyn_cast<IntegerType>(rhs.getElementType());
      if (!leftElement || !rightElement ||
          leftElement.getWidth() > element.getWidth() ||
          rightElement.getWidth() > element.getWidth())
        return;
      // Keep the narrow widening-dot family intact. A scalar contraction with
      // an accumulator-width input instead needs ordinary modular products
      // followed by a lane reduction; it has no surviving vector output axis.
      if (leftElement.getWidth() < element.getWidth() &&
          rightElement.getWidth() < element.getWidth())
        return;
      contracts.push_back(op);
    });
    IRRewriter b(&getContext());
    for (Operation *op : contracts) {
      Type result = op->getResult(0).getType();
      auto resultValue = dyn_cast<riscv::ValueType>(result);
      Type element = logicalElement(result);
      int64_t axis = op->getAttrOfType<DenseI64ArrayAttr>("over")[0];
      SmallVector<int64_t> shape, axes;
      for (Value input : op->getOperands()) {
        auto type = cast<riscv::ValueType>(input.getType());
        for (auto [id, extent] : llvm::zip(type.getAxisIds().asArrayRef(), type.getShape().asArrayRef()))
          if (!llvm::is_contained(axes, id)) { axes.push_back(id); shape.push_back(extent); }
      }
      auto position = llvm::find(axes, axis);
      if (position == axes.end()) continue;
      int64_t index = position - axes.begin();
      SmallVector<int64_t> freeAxes(axes), freeShape(shape);
      freeAxes.erase(freeAxes.begin() + index);
      freeShape.erase(freeShape.begin() + index);
      if (resultValue) {
        if (ArrayRef<int64_t>(freeAxes) != resultValue.getAxisIds().asArrayRef() ||
            ArrayRef<int64_t>(freeShape) != resultValue.getShape().asArrayRef()) continue;
      } else if (!freeAxes.empty()) {
        continue;
      }
      b.setInsertionPoint(op);
      auto productType = riscv::ValueType::get(b.getContext(), element,
          b.getDenseI64ArrayAttr(shape), b.getDenseI64ArrayAttr(axes), unassignedLayout(b, axes));
      auto promote = [&](Value input) -> Value {
        auto type = cast<riscv::ValueType>(input.getType());
        if (type.getElementType() == element) return input;
        auto promoted = riscv::ValueType::get(b.getContext(), element,
            type.getShape(), type.getAxisIds(),
            unassignedLayout(b, type.getAxisIds().asArrayRef()));
        Operation *conversion;
        if (cast<IntegerType>(type.getElementType()).getWidth() <
            cast<IntegerType>(element).getWidth())
          conversion = riscv::WidenOp::create(b, op->getLoc(), promoted, input,
                                             unselectedLeaf(b));
        else
          conversion = riscv::CastOp::create(b, op->getLoc(), promoted, input,
                                            unselectedLeaf(b));
        copyOrigin(op, conversion);
        return conversion->getResult(0);
      };
      // Widen each operand using its original signedness. Multiplication and
      // reduction then use the declared accumulator's modular integer width.
      Value lhs = promote(op->getOperand(0));
      Value rhs = promote(op->getOperand(1));
      auto product = riscv::BinaryOp::create(b, op->getLoc(), productType,
          lhs, rhs, "mul", unselectedLeaf(b));
      auto reduction = riscv::ReduceOp::create(b, op->getLoc(), result, product,
          "add", index, unselectedLeaf(b), ValueRange{});
      copyOrigin(op, product);
      copyOrigin(op, reduction);
      b.replaceOp(op, reduction.getResult());
    }
  }

  LogicalResult materializeScan(IRRewriter &b, riscv::ScanOp op, int64_t &nextAxis) {
    const unsigned axis = op.getAxis();
    auto input = cast<riscv::ValueType>(op.getInputs()[0].getType());
    int64_t extent = input.getShape()[axis];
    if (extent <= 0) return op.emitOpError("scan requires a static positive extent");
    b.setInsertionPoint(op);
    auto loc = op.getLoc();
    auto u32 = IntegerType::get(b.getContext(), 32, IntegerType::Unsigned);
    auto typeFor = [&](Type element, ArrayRef<int64_t> shape, ArrayRef<int64_t> axes) {
      return riscv::ValueType::get(b.getContext(), element, b.getDenseI64ArrayAttr(shape),
          b.getDenseI64ArrayAttr(axes), unassignedLayout(b, axes));
    };
    auto constant = [&](int64_t n) -> Value {
      return riscv::ConstantOp::create(b, loc, u32, b.getIntegerAttr(u32, n));
    };
    int64_t scanAxis = input.getAxisIds()[axis];
    auto indicesType = typeFor(u32, {extent}, {scanAxis});
    Value coordinates = riscv::IotaOp::create(b, loc, indicesType, 0, extent, unselectedLeaf(b));
    auto predicateType = typeFor(b.getI1Type(), {extent}, {scanAxis});
    SmallVector<Value> values(op.getInputs());
    auto &body = op.getCombine().front();
    // Inclusive doubling preserves tuple operand order. Reverse is exactly
    // flip(scan(flip(input))), matching Triton's noncommutative contract.
    for (int64_t distance = 1; distance < extent; distance *= 2) {
      Value delta = constant(distance);
      Value valid = riscv::CompareOp::create(b, loc, predicateType, coordinates,
          op.getReverse() ? constant(extent - distance) : delta,
          op.getReverse() ? "lt" : "ge", unselectedLeaf(b));
      Value shifted = riscv::BinaryOp::create(b, loc, indicesType, coordinates, delta,
          op.getReverse() ? "add" : "sub", unselectedLeaf(b));
      Value safe = riscv::SelectOp::create(b, loc, indicesType, valid, shifted,
          coordinates, unselectedLeaf(b));
      int64_t gatheredAxis = nextAxis++;
      SmallVector<int64_t> axes(input.getAxisIds().asArrayRef());
      axes[axis] = gatheredAxis;
      auto gatheredIndexType = typeFor(u32, {extent}, {gatheredAxis});
      Value index = riscv::ReshapeOp::create(b, loc, gatheredIndexType, safe,
          indicesType.getAxisIds(), DenseI64ArrayAttr());
      SmallVector<Attribute> selectors(input.getShape().size(), b.getStringAttr("all"));
      selectors[axis] = b.getStringAttr("gather");
      IRMapping mapping;
      for (auto [i, value] : llvm::enumerate(values)) {
        auto originalType = cast<riscv::ValueType>(value.getType());
        auto gatheredType = typeFor(originalType.getElementType(), input.getShape().asArrayRef(), axes);
        Value gathered = riscv::ExtractOp::create(b, loc, gatheredType, value,
            ValueRange{index}, b.getArrayAttr(selectors), unassignedAccess(b), unselectedLeaf(b));
        Value previous = riscv::ReshapeOp::create(b, loc, originalType, gathered,
            gatheredType.getAxisIds(), DenseI64ArrayAttr());
        mapping.map(body.getArgument(i), previous);
        mapping.map(body.getArgument(values.size() + i), value);
      }
      for (Operation &inner : body.without_terminator()) {
        SmallVector<Value> operands;
        bool shaped = false;
        for (Value operand : inner.getOperands()) {
          Value mapped = mapping.lookup(operand);
          operands.push_back(mapped);
          shaped |= isa<riscv::ValueType>(mapped.getType());
        }
        OperationState state(inner.getLoc(), inner.getName());
        state.addOperands(operands);
        Type element = inner.getResult(0).getType();
        state.addTypes(shaped ? Type(typeFor(element, input.getShape().asArrayRef(), input.getAxisIds().asArrayRef())) : element);
        state.addAttributes(inner.getAttrs());
        Operation *created = b.create(state);
        copyOrigin(op, created);
        mapping.map(inner.getResult(0), created->getResult(0));
      }
      for (auto [i, yielded] : llvm::enumerate(body.getTerminator()->getOperands()))
        values[i] = riscv::SelectOp::create(b, loc, values[i].getType(), valid,
            mapping.lookup(yielded), values[i], unselectedLeaf(b));
    }
    b.replaceOp(op, values);
    return success();
  }

  LogicalResult materialize(IRRewriter &b, riscv::ReduceRegionOp op, int64_t &nextAxis) {
    const unsigned axis = op.getAxis();
    auto inputType = cast<riscv::ValueType>(op.getInputs()[0].getType());
    int64_t extent = inputType.getShape()[axis];
    if (extent <= 0)
      return op.emitOpError("tuple reduction requires a static positive extent");
    b.setInsertionPoint(op);
    auto typeFor = [&](Type element, ArrayRef<int64_t> shape, ArrayRef<int64_t> axes) -> Type {
      if (shape.empty()) return element;
      return riscv::ValueType::get(b.getContext(), element, b.getDenseI64ArrayAttr(shape),
          b.getDenseI64ArrayAttr(axes), unassignedLayout(b, axes));
    };
    auto project = [&](Value value, Value index, StringRef selector,
                       ArrayRef<int64_t> shape, ArrayRef<int64_t> axes) -> Value {
      auto type = cast<riscv::ValueType>(value.getType());
      SmallVector<Attribute> selectors(type.getShape().size(), b.getStringAttr("all"));
      selectors[axis] = b.getStringAttr(selector);
      auto result = riscv::ExtractOp::create(b, op.getLoc(), typeFor(type.getElementType(), shape, axes),
          value, ValueRange{index}, b.getArrayAttr(selectors), unassignedAccess(b), unselectedLeaf(b));
      copyOrigin(op, result);
      return result;
    };
    SmallVector<Value> values(op.getInputs());
    SmallVector<int64_t> shape(inputType.getShape().asArrayRef()), axes(inputType.getAxisIds().asArrayRef());
    auto &body = op.getCombine().front();
    bool projected = false;
    while (extent > 1) {
      const int64_t inputExtent = extent;
      extent = llvm::divideCeil(inputExtent, int64_t{2});
      Value even, odd, tailIndex, paired;
      StringRef selector;
      if (extent == 1) {
        // The last pair has no pair axis: project its two fixed coordinates
        // directly, avoiding a degenerate gather and a subsequent reshape.
        shape.erase(shape.begin() + axis);
        axes.erase(axes.begin() + axis);
        even = arith::ConstantIndexOp::create(b, op.getLoc(), 0);
        odd = arith::ConstantIndexOp::create(b, op.getLoc(), 1);
        selector = "index";
        projected = true;
      } else {
        shape[axis] = extent;
        axes[axis] = nextAxis++;
        auto unsignedType = IntegerType::get(b.getContext(), 32, IntegerType::Unsigned);
        auto indexType = cast<riscv::ValueType>(typeFor(unsignedType, {extent}, {axes[axis]}));
        auto iota = riscv::IotaOp::create(b, op.getLoc(), indexType, 0, extent, unselectedLeaf(b));
        auto two = riscv::ConstantOp::create(b, op.getLoc(), unsignedType, b.getIntegerAttr(unsignedType, 2));
        auto one = riscv::ConstantOp::create(b, op.getLoc(), unsignedType, b.getIntegerAttr(unsignedType, 1));
        even = riscv::BinaryOp::create(b, op.getLoc(), indexType, iota, two, "mul", unselectedLeaf(b));
        odd = riscv::BinaryOp::create(b, op.getLoc(), indexType, even, one, "add", unselectedLeaf(b));
        if (inputExtent % 2) {
          // Carry the unpaired final tuple to the next tree level. Evaluate
          // its inactive combiner lane on an existing pair, so neither an
          // out-of-bounds read nor an invented numerical identity is needed.
          tailIndex = even;
          auto bound = riscv::ConstantOp::create(b, op.getLoc(), unsignedType,
              b.getIntegerAttr(unsignedType, inputExtent / 2));
          auto predicateType = typeFor(b.getI1Type(), {extent}, {axes[axis]});
          paired = riscv::CompareOp::create(b, op.getLoc(), predicateType, iota, bound,
              "lt", unselectedLeaf(b));
          auto zero = riscv::ConstantOp::create(b, op.getLoc(), unsignedType,
              b.getIntegerAttr(unsignedType, 0));
          even = riscv::SelectOp::create(b, op.getLoc(), indexType, paired, even, zero, unselectedLeaf(b));
          odd = riscv::SelectOp::create(b, op.getLoc(), indexType, paired, odd, one, unselectedLeaf(b));
        }
        for (Operation *created : {iota.getOperation(), two.getOperation(), one.getOperation()}) copyOrigin(op, created);
        selector = "gather";
      }
      copyOrigin(op, even.getDefiningOp());
      copyOrigin(op, odd.getDefiningOp());
      IRMapping mapping;
      for (auto [i, value] : llvm::enumerate(values)) {
        mapping.map(body.getArgument(i), project(value, even, selector, shape, axes));
        mapping.map(body.getArgument(values.size() + i), project(value, odd, selector, shape, axes));
      }
      for (Operation &inner : body.without_terminator()) {
        SmallVector<Value> operands;
        bool shaped = false;
        for (Value input : inner.getOperands()) {
          Value mapped = mapping.lookup(input);
          operands.push_back(mapped);
          shaped |= isa<riscv::ValueType>(mapped.getType());
        }
        OperationState state(inner.getLoc(), inner.getName());
        state.addOperands(operands);
        Type scalar = inner.getResult(0).getType();
        state.addTypes(shaped ? typeFor(scalar, shape, axes) : scalar);
        state.addAttributes(inner.getAttrs());
        auto *created = b.create(state);
        copyOrigin(op, created);
        mapping.map(inner.getResult(0), created->getResult(0));
      }
      SmallVector<Value> nextValues;
      for (auto [i, yielded] : llvm::enumerate(body.getTerminator()->getOperands())) {
        Value value = mapping.lookup(yielded);
        if (!shape.empty() && !isa<riscv::ValueType>(value.getType()))
          return op.emitOpError("tuple combiner result independent of all tuple operands is not supported");
        if (paired) {
          Value tail = project(values[i], tailIndex, "gather", shape, axes);
          value = riscv::SelectOp::create(b, op.getLoc(), value.getType(), paired,
              value, tail, unselectedLeaf(b));
          copyOrigin(op, value.getDefiningOp());
        }
        nextValues.push_back(value);
      }
      values = std::move(nextValues);
    }
    if (!projected) {
      shape.erase(shape.begin() + axis);
      axes.erase(axes.begin() + axis);
      auto zero = arith::ConstantIndexOp::create(b, op.getLoc(), 0);
      copyOrigin(op, zero);
      for (Value &value : values) value = project(value, zero, "index", shape, axes);
    }
    b.replaceOp(op, values);
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> weft::createMaterializeRISCVReductionsPass() {
  return std::make_unique<MaterializeRISCVReductionsPass>();
}
