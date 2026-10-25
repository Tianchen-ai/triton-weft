#include "Weft/Target/FullTiles.h"
#include "Weft/Target/RISCVPasses.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace {
class SeparateFullTiles : public PassWrapper<SeparateFullTiles, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SeparateFullTiles)
  StringRef getArgument() const override { return "weft-riscv-separate-full-tiles"; }
  StringRef getDescription() const override {
    return "Prove complete physical iterations while retaining ordered Level tails and carries";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    bool invalid = false;
    getOperation().walk([&](weft::riscv::KernelOp kernel) {
      if (kernel.getResourcesMaterialized()) {
        kernel.emitError("complete-tile separation requires open physical resources");
        invalid = true; return;
      }
      auto growth = weft::separateFullTiles(kernel, true);
      if (failed(growth)) { invalid = true; return; }
      Builder b(&getContext());
      kernel->setAttr("weft.riscv.full_tile_operation_budget", b.getI64IntegerAttr(growth->budget));
      kernel->setAttr("weft.riscv.full_tile_added_operations", b.getI64IntegerAttr(growth->operations));
      kernel->setAttr("weft.riscv.full_domain_versions", b.getI64IntegerAttr(growth->domainVersions));
    });
    if (invalid) signalPassFailure();
  }
};
}
std::unique_ptr<Pass> weft::createSeparateRISCVFullTilesPass() {
  return std::make_unique<SeparateFullTiles>();
}
