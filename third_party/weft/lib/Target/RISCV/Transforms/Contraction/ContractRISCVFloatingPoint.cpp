#include "Transforms/Support/RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/FMAIssue.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "Weft/Target/RISCV/RISCVPasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
        const StringRef kind = add.getKind() == "add" ? "add" :
            operand == 0 ? "subtract_accumulator" : "subtract_product";
        // A loop-invariant scalar addend can have one explicit broadcast before
        // a complete, repeated issue loop. vfmadd overwrites the dying vector
        // multiplicand and leaves that broadcast reusable. Unlike accumulator-
        // tied vfmacc this requires no copy for each iteration.
        scf::ForOp broadcastLoop;
        const bool scalarAddend = shaped && accumulator.getType() == element;
        std::optional<int64_t> broadcastGroups;
        if (scalarAddend) {
          broadcastGroups = riscv::sharedSplatRegisterGroups(shaped);
          broadcastLoop = dyn_cast<scf::ForOp>(add->getParentOp());
          auto trips = broadcastLoop ? analysis::staticTripCount(broadcastLoop) : std::nullopt;
          if (!trips || *trips < 2 || !broadcastLoop.isDefinedOutsideOfLoop(accumulator))
            broadcastLoop = {};
          auto parts = riscv::rvvCarrierCount(shaped.getLayout());
          if (kind != "add" || !broadcastGroups || !parts ||
              (!broadcastLoop && *parts < 2)) {
            ++scalarAddendsRetained;
            continue;
          }
        } else if (accumulator.getType() != result) continue;
        auto instruction = riscv::pointwiseFMAInstruction(lhs.getType(), rhs.getType(), result, kind, scalarAddend);
        if (instruction.empty()) {
          std::swap(lhs, rhs);
          instruction = riscv::pointwiseFMAInstruction(lhs.getType(), rhs.getType(), result, kind, scalarAddend);
        }
        if (instruction.empty()) continue;
        if (scalarAddend && (!lhs.hasOneUse() || registerStorageIdentity(lhs) != lhs)) {
          ++scalarAddendsRetained;
          continue;
        }
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
        if (scalarAddend) {
          resident += *broadcastGroups;
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
        if (scalarAddend) {
          b.setInsertionPoint(broadcastLoop ? broadcastLoop.getOperation() : add.getOperation());
          auto splat = riscv::RVVSplatOp::create(b, add.getLoc(), shaped, accumulator,
              leaf(b, "rvv", "splat", "rvv.splat.shared", "rvv.splat.shared", 0,
                   *broadcastGroups, 0, 0, "none", "exact"));
          copyOrigin(add, splat);
          accumulator = splat.getResult();
          ++reusedScalarAddends;
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
  Statistic reusedScalarAddends{this, "weft-fma-shared-scalar-addends",
                               "Explicit loop-invariant broadcasts consumed by multiplicand-tied FMA"};
  Statistic resourceFrontiersRetained{this, "weft-fma-resource-frontiers-retained",
                                      "Unfused schedules retained to keep simultaneous input residency legal"};
};
} // namespace

std::unique_ptr<Pass> weft::createContractRISCVFloatingPointPass() {
  return std::make_unique<ContractRISCVFloatingPointPass>();
}
