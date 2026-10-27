#include "Weft/Target/RISCV/RISCVPasses.h"
#include "Transforms/Control/PipelineSchedule.h"
#include "Transforms/Control/PipelineShapeBindings.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include <functional>

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
void copyLoopAttributes(scf::ForOp source, Operation *target, bool levelIdentity) {
  for (NamedAttribute attribute : source->getAttrs()) {
    auto name = attribute.getName();
    if (name == "weft.riscv.schedule" || name == "weft.riscv.pipeline_peak") continue;
    if (!levelIdentity && (name == "weft.riscv.level" || name == "weft.riscv.validity_slots")) continue;
    if (!isa<scf::ForOp>(target) &&
        (name == "weft.riscv.unroll_factor" || name == "weft.riscv.unroll_order")) continue;
    target->setAttr(name, attribute.getValue());
  }
  if (!levelIdentity && !target->hasAttr("weft.riscv.direction"))
    if (auto level = source->getAttrOfType<riscv::LevelAttr>("weft.riscv.level"))
      target->setAttr("weft.riscv.direction", StringAttr::get(target->getContext(), level.getDirection()));
}

} // namespace

FailureOr<PipelineExpansion> riscv_internal::expandPipeline(
    const PipelineSchedule &schedule, IRRewriter &b, int64_t &shapeIdentity) {
  if (failed(schedule.verify()) || failed(verifyPipelineShapeStages(schedule.loop))) return failure();
  return expandPhysicalPipeline(schedule, b, [&](IRRewriter &rewriter, Operation *source, IRMapping &values) {
    return clonePipelineOperation(rewriter, source, values, shapeIdentity);
  }, copyLoopAttributes,
  [&](IRRewriter &rewriter, const PhysicalPipelineBuffer &buffer) -> FailureOr<Value> {
    auto original = cast<riscv::LocalType>(buffer.allocation->getResult(0).getType());
    int64_t id = ++shapeIdentity;
    auto type = riscv::LocalType::get(rewriter.getContext(), rewriter.getI8Type(),
        rewriter.getDenseI64ArrayAttr({}), rewriter.getDenseI64ArrayAttr({}),
        buffer.bytes * buffer.versions(), buffer.alignment, id, "pipeline",
        original.getOwnerDomainId(), id, original.getLifetimeEndDomainId(), "versioned-local");
    auto loc = buffer.allocation->getLoc();
    auto size = arith::ConstantIndexOp::create(rewriter, loc, type.getSizeBytes());
    return riscv::LocalAllocOp::create(rewriter, loc, type, size).getResult();
  }, [](IRRewriter &rewriter, const PhysicalPipelineBuffer &buffer,
        Value storage, Value slot) -> FailureOr<Value> {
    return riscv::LocalSlotOp::create(rewriter, buffer.allocation->getLoc(),
        buffer.allocation->getResult(0).getType(), storage, slot,
        rewriter.getI64IntegerAttr(buffer.bytes), rewriter.getI64IntegerAttr(buffer.versions())).getResult();
  });
}

namespace {
class PipelineRISCVLevelsPass
    : public PassWrapper<PipelineRISCVLevelsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineRISCVLevelsPass)
  StringRef getArgument() const override { return "weft-riscv-pipeline-levels"; }
  StringRef getDescription() const override {
    return "Expand a verified physical stage/order into versioned prologue, steady state and epilogue";
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    int64_t identity = nextPipelineShapeIdentity(getOperation());
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.schedule")) loops.push_back(loop);
    });
    for (auto loop : loops) {
      auto binding = loop->getAttrOfType<riscv::ScheduleAttr>("weft.riscv.schedule");
      if (binding.getPipelineDepth() == 1) {
        loop->removeAttr("weft.riscv.schedule");
        continue;
      }
      auto schedule = PipelineSchedule::read(loop);
      if (failed(schedule)) { signalPassFailure(); return; }
      auto expanded = expandPipeline(*schedule, b, identity);
      if (failed(expanded)) { signalPassFailure(); return; }
      b.replaceOp(loop, expanded->guard.getResults());
    }
    bool remaining = false;
    getOperation().walk([&](Operation *operation) {
      if (operation->hasAttr("weft.riscv.schedule") || operation->hasAttr(pipelineStage) ||
          operation->hasAttr(pipelineOrder) || operation->hasAttr(pipelineInvariant) || operation->hasAttr("schedule")) {
        operation->emitError("physical schedule survived without structural pipeline expansion");
        remaining = true;
      }
    });
    if (remaining) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> weft::createPipelineRISCVLevelsPass() {
  return std::make_unique<PipelineRISCVLevelsPass>();
}
