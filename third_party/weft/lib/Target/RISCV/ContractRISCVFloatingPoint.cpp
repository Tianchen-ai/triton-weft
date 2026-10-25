#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/FMAIssue.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
class ContractRISCVFloatingPointPass
    : public PassWrapper<ContractRISCVFloatingPointPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContractRISCVFloatingPointPass)
  ContractRISCVFloatingPointPass() = default;
  ContractRISCVFloatingPointPass(const ContractRISCVFloatingPointPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override { return "weft-riscv-contract-floating-point"; }
  StringRef getDescription() const override {
    return "Contract local single-use floating multiply/add/subtract into typed FMA operations";
  }
  void runOnOperation() override {
    SmallVector<riscv::BinaryOp> additions;
    getOperation().walk([&](riscv::BinaryOp op) {
      if (op.getKind() == "add" || op.getKind() == "sub") additions.push_back(op);
    });
    IRRewriter b(&getContext());
    for (auto add : additions) {
      for (unsigned operand = 0; operand < 2; ++operand) {
        auto product = add->getOperand(operand).getDefiningOp<riscv::BinaryOp>();
        if (!product || product.getKind() != "mul" ||
            !product.getResult().hasOneUse() || product->getBlock() != add->getBlock() ||
            product.getResult().getType() != add.getResult().getType()) continue;
        Value lhs = product.getLhs(), rhs = product.getRhs();
        Value accumulator = add->getOperand(1 - operand);
        Type result = add.getResult().getType();
        auto shaped = dyn_cast<riscv::ValueType>(result);
        Type element = shaped ? shaped.getElementType() : result;
        if (!element.isF32() && !element.isF64()) continue;
        // The selected FMA ties its output to the vector accumulator. A scalar
        // addend therefore needs a splat (or a copy of a shared splat) for every
        // packet, even when the scalar itself is loop-invariant. This replaces
        // mul + .vf add/sub with splat/copy + FMA: equal issue work with longer
        // vector residency. Preserve the existing implementation on that tie;
        // do not price a broadcast hoist or a different tied operand that this
        // materializer does not actually produce.
        if (accumulator.getType() != result) {
          if (accumulator.getType() == element) ++scalarAddendsRetained;
          continue;
        }
        const StringRef kind = add.getKind() == "add" ? "add" :
            operand == 0 ? "subtract_accumulator" : "subtract_product";
        auto instruction = riscv::pointwiseFMAInstruction(lhs.getType(), rhs.getType(), result, kind);
        if (instruction.empty()) {
          std::swap(lhs, rhs);
          instruction = riscv::pointwiseFMAInstruction(lhs.getType(), rhs.getType(), result, kind);
        }
        if (instruction.empty()) continue;
        // Fusion makes all three inputs simultaneously necessary. The unfused
        // pair can reload the addend after the multiply's inputs die. Preserve
        // that legal schedule when even the fused input frontier exceeds the
        // target; spilling one of these immediate operands cannot repair it.
        llvm::DenseSet<Value> storage;
        int64_t resident = 0;
        bool known = true;
        for (Value input : {lhs, rhs, accumulator}) {
          if (!storage.insert(registerStorageIdentity(input)).second) continue;
          auto groups = residentRegisterGroups(input);
          if (!groups) { known = false; break; }
          resident += *groups;
        }
        if (!known || resident > add->getParentOfType<riscv::KernelOp>().getTarget().getVectorRegisters()) {
          ++resourceFrontiersRetained;
          continue;
        }
        bool crossesEffect = false;
        for (Operation *op = product->getNextNode(); op != add; op = op->getNextNode())
          if (op->getNumRegions() || !riscv::hasOnlyReadEffects(op)) {
            crossesEffect = true;
            break;
          }
        if (crossesEffect) continue;
        if (add->getParentOfType<riscv::KernelOp>().getResourcesMaterialized()) {
          add.emitError("floating contraction requires an open physical resource boundary");
          signalPassFailure();
          return;
        }
        b.setInsertionPoint(add);
        auto fused = riscv::FMAOp::create(b, add.getLoc(), result, lhs, rhs, accumulator,
            leaf(b, isa<riscv::ValueType>(result) ? "rvv" : "scalar", "fma", instruction, instruction,
                 issueRegisterGroups(lhs) + issueRegisterGroups(rhs) + issueRegisterGroups(accumulator),
                 issueRegisterGroups(add.getResult()), 0, 0, "none",
                 isa<riscv::ValueType>(result) ? "exact" : "agnostic"), kind);
        copyOrigin(add, fused);
        b.replaceOp(add, fused.getResult());
        b.eraseOp(product);
        ++contracted;
        break;
      }
    }
  }
private:
  Statistic contracted{this, "weft-pointwise-fma", "Single-use floating multiply/add pairs contracted"};
  Statistic scalarAddendsRetained{this, "weft-fma-scalar-addends-retained",
                                  "Scalar addends retained after accounting for broadcast/copy issue work"};
  Statistic resourceFrontiersRetained{this, "weft-fma-resource-frontiers-retained",
                                      "Unfused schedules retained to keep simultaneous input residency legal"};
};
} // namespace

std::unique_ptr<Pass> weft::createContractRISCVFloatingPointPass() {
  return std::make_unique<ContractRISCVFloatingPointPass>();
}
