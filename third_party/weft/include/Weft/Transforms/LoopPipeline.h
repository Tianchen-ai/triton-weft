#ifndef WEFT_TRANSFORMS_LOOP_PIPELINE_H
#define WEFT_TRANSFORMS_LOOP_PIPELINE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace weft {

// One compiler-owned allocation. Descriptor operations can be replayed from a
// selected slot; payload accesses keep their selected stage and memory order.
struct PhysicalPipelineBuffer {
  mlir::Operation *allocation = nullptr;
  llvm::SmallVector<mlir::Operation *> descriptors;
  llvm::SmallVector<mlir::Operation *> accesses;
  unsigned firstStage = 0, lastStage = 0;
  uint64_t bytes = 0, alignment = 1;

  unsigned versions() const { return lastStage - firstStage + 1; }
};

// A transient view of selected operations in one physical loop. Targets verify
// their effects and ownership and write stage/order into IR before expansion.
struct PhysicalPipeline {
  mlir::scf::ForOp loop;
  llvm::SmallVector<mlir::Operation *> producers, consumers, ordered, invariants;
  llvm::SmallVector<mlir::Value> transported;
  llvm::DenseMap<mlir::Operation *, unsigned> stages;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>> orderedAfter;
  llvm::SmallVector<PhysicalPipelineBuffer> buffers;
  bool transportInduction = false;
  bool descending = false;
  unsigned depth = 2;

  bool isProducer(mlir::Operation *op) const { return llvm::is_contained(producers, op); }
  bool isInvariant(mlir::Operation *op) const { return llvm::is_contained(invariants, op); }
  unsigned stage(mlir::Operation *op) const {
    auto found = stages.find(op);
    return found == stages.end() ? (isProducer(op) ? 0 : depth - 1) : found->second;
  }
  bool isBufferDescriptor(mlir::Operation *op) const {
    return llvm::any_of(buffers, [&](const auto &buffer) {
      return buffer.allocation == op || llvm::is_contained(buffer.descriptors, op);
    });
  }
};

struct PhysicalPipelineExpansion {
  mlir::scf::IfOp guard;
  mlir::scf::ForOp steady;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> steadyOperations;
};

using PipelineClone = llvm::function_ref<mlir::Operation *(
    mlir::IRRewriter &, mlir::Operation *, mlir::IRMapping &)>;
using PipelineLoopAttributes = llvm::function_ref<void(
    mlir::scf::ForOp, mlir::Operation *, bool)>;
using PipelineDependencies = llvm::function_ref<mlir::FailureOr<
    llvm::SmallVector<mlir::Value>>(mlir::Operation *)>;
using PipelineBufferAllocator = llvm::function_ref<mlir::FailureOr<mlir::Value>(
    mlir::IRRewriter &, const PhysicalPipelineBuffer &)>;
using PipelineBufferSelector = llvm::function_ref<mlir::FailureOr<mlir::Value>(
    mlir::IRRewriter &, const PhysicalPipelineBuffer &, mlir::Value, mlir::Value)>;

void addPipelineMemoryOrder(PhysicalPipeline &schedule,
                           llvm::ArrayRef<mlir::Operation *> accesses);
void closePhysicalPipelineBuffers(PhysicalPipeline &schedule);

// Subdivide the existing independent producer frontier along its read/use-def
// chain. Ownership/shape dependencies that cannot travel remain in one stage.
// The author's recurrence remains entirely in the last stage.
mlir::LogicalResult distributePhysicalPipelineStages(
    PhysicalPipeline &schedule, PipelineDependencies dependencies,
    llvm::function_ref<bool(mlir::Operation *)> reads,
    llvm::function_ref<bool(mlir::Value)> transportable);

// Verify the total order and all SSA/control captures before expansion. Target
// callbacks add typed coordinate dependencies without duplicating graph rules.
mlir::LogicalResult verifyPhysicalPipeline(const PhysicalPipeline &schedule,
                                          PipelineDependencies dependencies);

// The source remains alive for candidate evaluation; the caller installs the
// selected guard. Numerical carries are updated once per original iteration.
mlir::FailureOr<PhysicalPipelineExpansion> expandPhysicalPipeline(
    const PhysicalPipeline &schedule, mlir::IRRewriter &rewriter,
    PipelineClone clone, PipelineLoopAttributes attributes,
    PipelineBufferAllocator allocate = nullptr,
    PipelineBufferSelector select = nullptr);

} // namespace weft
#endif
