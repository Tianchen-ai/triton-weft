#ifndef WEFT_TARGET_LOOP_PIPELINE_H
#define WEFT_TARGET_LOOP_PIPELINE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace weft {

// A transient view of selected operations in one physical loop. Targets verify
// their effects and ownership and write stage/order into IR before expansion.
struct PhysicalPipeline {
  mlir::scf::ForOp loop;
  llvm::SmallVector<mlir::Operation *> producers, consumers, ordered, invariants;
  llvm::SmallVector<mlir::Value> transported;
  bool transportInduction = false;
  bool descending = false;
  unsigned depth = 2;

  bool isProducer(mlir::Operation *op) const { return llvm::is_contained(producers, op); }
  bool isInvariant(mlir::Operation *op) const { return llvm::is_contained(invariants, op); }
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

// Verify the total order and all SSA/control captures before expansion. Target
// callbacks add typed coordinate dependencies without duplicating graph rules.
mlir::LogicalResult verifyPhysicalPipeline(const PhysicalPipeline &schedule,
                                          PipelineDependencies dependencies);

// The source remains alive for candidate evaluation; the caller installs the
// selected guard. Numerical carries are updated once per original iteration.
mlir::FailureOr<PhysicalPipelineExpansion> expandPhysicalPipeline(
    const PhysicalPipeline &schedule, mlir::IRRewriter &rewriter,
    PipelineClone clone, PipelineLoopAttributes attributes);

} // namespace weft
#endif
