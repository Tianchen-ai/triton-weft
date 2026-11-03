#include "Weft/Dialect/Kernel/Transforms/SourceSpecialization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
class NormalizeCanonicalControl
    : public PassWrapper<NormalizeCanonicalControl, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeCanonicalControl)
  StringRef getArgument() const override { return "weft-normalize-canonical-control"; }
  StringRef getDescription() const override {
    return "Normalize ordinary SCF carries before choosing their physical representations";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    SmallVector<Operation *> loops;
    getOperation().walk([&](scf::ForOp loop) { loops.push_back(loop); });
    GreedyRewriteConfig configuration;
    configuration.setStrictness(GreedyRewriteStrictness::ExistingOps);
    if (failed(applyOpPatternsGreedily(loops, std::move(patterns), configuration))) signalPassFailure();
  }
};
}

std::unique_ptr<Pass> weft::createNormalizeCanonicalControlPass() {
  return std::make_unique<NormalizeCanonicalControl>();
}
