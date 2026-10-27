#ifndef WEFT_TARGET_PIPELINE_SCHEDULE_H
#define WEFT_TARGET_PIPELINE_SCHEDULE_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Transforms/LoopPipeline.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace weft::riscv_internal {

inline constexpr llvm::StringLiteral pipelineStage = "weft.riscv.pipeline_stage";
inline constexpr llvm::StringLiteral pipelineOrder = "weft.riscv.pipeline_order";
inline constexpr llvm::StringLiteral pipelineInvariant = "weft.riscv.pipeline_invariant";

// A query over one physical loop, never a persistent planning IR. Stage and
// total steady-state order are written on the operations themselves. Expansion
// reads those decisions back and validates them before making any copies.
struct PipelineSchedule : PhysicalPipeline {
  mlir::LogicalResult verify() const;
  void write() const;
  static mlir::FailureOr<PipelineSchedule> read(mlir::scf::ForOp loop);
};

// No split is a normal absence of an eligible producer/consumer cluster.
// Invalid effects, bindings or dependency contracts are diagnostics, not an
// invitation to retry another implementation of the source program.
enum class PipelineFrontier { Contraction, Carry };
mlir::FailureOr<std::optional<PipelineSchedule>> schedulePipeline(
    mlir::scf::ForOp loop, PipelineFrontier frontier = PipelineFrontier::Contraction);

using PipelineExpansion = PhysicalPipelineExpansion;

mlir::FailureOr<PipelineExpansion> expandPipeline(
    const PipelineSchedule &schedule, mlir::IRRewriter &rewriter,
    int64_t &shapeIdentity);

// Reorders the two already selected stage chains on a disposable expanded
// physical loop. The returned order refers to the original scheduled ops.
// Both stages retain their own dependency/effect order; final residency uses
// the same region-aware analysis as ordinary resource closure.
mlir::FailureOr<llvm::SmallVector<mlir::Operation *>> orderPipelineFrontiers(
    const PipelineSchedule &schedule, const PipelineExpansion &expansion,
    int64_t registerBudget);

// The same lifetime frontier also orders an explicitly unrolled read-only
// issue window. Priorities are the selected operation/iteration order; actual
// SSA and local-shape dependencies remain hard constraints.
mlir::FailureOr<llvm::SmallVector<mlir::Operation *>> orderReadOnlyIssues(
    mlir::Operation *owner, mlir::Block &body,
    llvm::ArrayRef<mlir::Operation *> operations,
    llvm::ArrayRef<std::pair<int64_t, int64_t>> priorities,
    int64_t registerBudget);

// Reuse the issue scheduler's scalar frontier when bounding native expansion.
// Only flat loops have an independent integer issue estimate here.
inline constexpr int64_t nativeIntegerRegisterBudget = 27; // Excludes x0, sp, gp, tp, ra.
struct NativeIntegerIssue {
  int64_t peak = 0;
  int64_t recurrences = 0;
  bool fits(int64_t copies) const {
    return copies >= 1 && recurrences >= 0 && recurrences <= peak &&
        recurrences <= nativeIntegerRegisterBudget &&
        peak - recurrences <= (nativeIntegerRegisterBudget - recurrences) / copies;
  }
};
mlir::FailureOr<std::optional<NativeIntegerIssue>> nativeIntegerIssuePeak(mlir::scf::ForOp loop);

} // namespace weft::riscv_internal
#endif
