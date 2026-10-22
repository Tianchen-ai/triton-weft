#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/FMAIssue.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"

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
        auto vector = dyn_cast<riscv::ValueType>(result);
        const bool scalarAccumulator = vector && accumulator.getType() == vector.getElementType();
        bool reusableSplat = false;
        if (scalarAccumulator) {
          for (Operation *parent = add->getParentOp(); parent; parent = parent->getParentOp())
            if (auto loop = dyn_cast<LoopLikeOpInterface>(parent)) {
              reusableSplat = loop.isDefinedOutsideOfLoop(accumulator);
              break;
            }
        }
        // A changing scalar addend already has a direct .vf add/subtract form;
        // splat + FMA saves no instruction. An invariant addend can amortize
        // its broadcast across loop iterations, so keep that fusion available.
        if (accumulator.getType() != result && !reusableSplat) continue;
        const StringRef kind = add.getKind() == "add" ? "add" :
            operand == 0 ? "subtract_accumulator" : "subtract_product";
        auto instruction = riscv::pointwiseFMAInstruction(lhs.getType(), rhs.getType(), result, kind);
        if (instruction.empty()) {
          std::swap(lhs, rhs);
          instruction = riscv::pointwiseFMAInstruction(lhs.getType(), rhs.getType(), result, kind);
        }
        if (instruction.empty()) continue;
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
        if (scalarAccumulator) {
          auto splat = riscv::RVVSplatOp::create(b, add.getLoc(), vector, accumulator,
              leaf(b, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
                   vector.getLayout().getRegisterGroups()));
          copyOrigin(add, splat);
          accumulator = splat.getResult();
        }
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
};
} // namespace

std::unique_ptr<Pass> weft::createContractRISCVFloatingPointPass() {
  return std::make_unique<ContractRISCVFloatingPointPass>();
}
