#include "Transforms/Arithmetic/NumericConversions.h"
#include "Transforms/Support/RISCVPhysicalSupport.h"
#include "Weft/Target/RISCV/RISCVPasses.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

LogicalResult riscv_internal::materializeNumericConversions(ModuleOp module, IRRewriter &b) {
  auto inherit = [](Operation *source, Operation *target) {
    for (StringRef name : {"source_origin", "canonical_op"})
      if (auto value = source->getAttr(name)) target->setAttr(name, value);
  };
  SmallVector<Operation *> operations;
  module.walk([&](Operation *op) {
    if (isa<riscv::CastOp, riscv::WidenOp>(op)) operations.push_back(op);
  });
  for (Operation *op : operations) {
    auto source = dyn_cast<riscv::ValueType>(op->getOperand(0).getType());
    auto result = dyn_cast<riscv::ValueType>(op->getResult(0).getType());
    if (!source || !result || result.getLayout().getCarrier() != "rvv") continue;
    if (source.getElementType().isF16() && result.getElementType().isF64()) {
      // RVV widens floating elements by one machine width per instruction.
      // Both steps are exact, including subnormals and signed zero; the
      // intermediate retains the source coordinates and active domain.
      b.setInsertionPoint(op);
      auto intermediate = riscv::ValueType::get(b.getContext(), b.getF32Type(),
          source.getShape(), source.getAxisIds(), source.getLayout());
      auto layout = projectLayout(b, intermediate, source.getLayout(),
          op->getParentOfType<riscv::KernelOp>().getTarget());
      if (!layout) return op->emitError("floating widening has no legal intermediate layout");
      intermediate = cast<riscv::ValueType>(withLayout(intermediate, layout));
      Value wide = riscv::WidenOp::create(b, op->getLoc(), intermediate,
          op->getOperand(0), unselectedLeaf(b));
      Value converted = riscv::WidenOp::create(b, op->getLoc(), result, wide, unselectedLeaf(b));
      inherit(op, wide.getDefiningOp());
      inherit(op, converted.getDefiningOp());
      b.replaceOp(op, converted);
      continue;
    }
    auto integer = dyn_cast<IntegerType>(source.getElementType());
    if (!integer) continue;
    b.setInsertionPoint(op);
    Type element = result.getElementType();
    if (integer.getWidth() == 1 && !integer.isUnsigned() && !element.isInteger(1)) {
      // A signless predicate conversion must explicitly select numeric 0/1.
      // Encoding ui1 values already use a normalized unsigned byte carrier;
      // retain their ordinary zero extension, without a byte->mask->value
      // round trip. Selected packed masks have explicit ingress/egress ops.
      auto constant = [&](int value) -> Value {
        Attribute attr = isa<FloatType>(element) ? Attribute(b.getFloatAttr(element, value))
                                                 : Attribute(b.getIntegerAttr(element, value));
        return riscv::ConstantOp::create(b, op->getLoc(), element, attr);
      };
      Value selected = riscv::SelectOp::create(b, op->getLoc(), result, op->getOperand(0),
          constant(1), constant(0), unselectedLeaf(b));
      inherit(op, selected.getDefiningOp());
      b.replaceOp(op, selected);
      continue;
    }
    if (integer.getWidth() <= 8 && result.getElementType().isF32()) {
      auto element = IntegerType::get(b.getContext(), 32,
          integer.isSigned() ? IntegerType::Signed : IntegerType::Unsigned);
      auto wideType = riscv::ValueType::get(b.getContext(), element, result.getShape(), result.getAxisIds(), result.getLayout());
      Value wide = riscv::WidenOp::create(b, op->getLoc(), wideType, op->getOperand(0), unselectedLeaf(b));
      Value converted = riscv::CastOp::create(b, op->getLoc(), result, wide, unselectedLeaf(b));
      inherit(op, wide.getDefiningOp()); inherit(op, converted.getDefiningOp());
      b.replaceOp(op, converted);
      continue;
    }
    auto target = dyn_cast<IntegerType>(result.getElementType());
    if (!isa<riscv::CastOp>(op) || !target || target.getWidth() < 2 || target.getWidth() >= integer.getWidth()) continue;
    const unsigned carrierWidth = std::max(8u, target.getWidth());
    if ((integer.getWidth() != 16 && integer.getWidth() != 32 && integer.getWidth() != 64) ||
        (carrierWidth != 8 && carrierWidth != 16 && carrierWidth != 32)) continue;
    Value value = op->getOperand(0);
    unsigned width = integer.getWidth();
    while (width > carrierWidth) {
      width /= 2;
      auto element = IntegerType::get(b.getContext(), width, integer.getSignedness());
      auto type = riscv::ValueType::get(b.getContext(), element, result.getShape(), result.getAxisIds(), source.getLayout());
      auto layout = projectLayout(b, type, source.getLayout(), op->getParentOfType<riscv::KernelOp>().getTarget());
      if (!layout) return op->emitError("integer truncation has no legal intermediate layout");
      type = cast<riscv::ValueType>(withLayout(type, layout));
      Value narrowed = riscv::NarrowOp::create(b, op->getLoc(), type, value, "none", false, unselectedLeaf(b));
      inherit(op, narrowed.getDefiningOp());
      value = narrowed;
    }
    if (value.getType() != result) {
      value = riscv::CastOp::create(b, op->getLoc(), result, value, unselectedLeaf(b));
      inherit(op, value.getDefiningOp());
    }
    b.replaceOp(op, value);
  }
  return success();
}

namespace {
class MaterializeRISCVNumericConversionsPass
    : public PassWrapper<MaterializeRISCVNumericConversionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVNumericConversionsPass)
  StringRef getArgument() const override { return "weft-riscv-materialize-numeric-conversions"; }
  StringRef getDescription() const override { return "Close machine-width conversion steps exposed by physical value programs"; }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    if (failed(materializeNumericConversions(getOperation(), b))) signalPassFailure();
  }
};
}
std::unique_ptr<Pass> weft::createMaterializeRISCVNumericConversionsPass() {
  return std::make_unique<MaterializeRISCVNumericConversionsPass>();
}
