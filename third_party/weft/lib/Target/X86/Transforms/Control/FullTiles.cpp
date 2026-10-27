#include "Weft/Transforms/FullTiles.h"
#include "Weft/Target/X86/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace {
class SeparateFullTiles : public PassWrapper<SeparateFullTiles, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SeparateFullTiles)
  StringRef getArgument() const override { return "weft-x86-separate-full-tiles"; }
  StringRef getDescription() const override {
    return "Separate complete tile iterations from the masked remainder without changing traversal or effects";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    bool invalid = false;
    getOperation().walk([&](func::FuncOp function) {
      if (function.isExternal()) return;
      auto growth = weft::separateFullTiles(function, false);
      if (failed(growth)) { invalid = true; return; }
      Builder b(&getContext());
      function->setAttr("weft.x86.full_tile_operation_budget", b.getI64IntegerAttr(growth->budget));
      function->setAttr("weft.x86.full_tile_added_operations", b.getI64IntegerAttr(growth->operations));
      function->setAttr("weft.x86.full_domain_versions", b.getI64IntegerAttr(growth->domainVersions));
    });
    if (invalid) signalPassFailure();
  }
};
}
std::unique_ptr<Pass> weft::createX86FullTilePass() {
  return std::make_unique<SeparateFullTiles>();
}
