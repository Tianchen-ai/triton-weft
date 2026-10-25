#ifndef WEFT_ANALYSIS_PHYSICAL_DEPENDENCE_H
#define WEFT_ANALYSIS_PHYSICAL_DEPENDENCE_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace weft::analysis {
// Operands captured by a complete structured unit, excluding values defined
// inside its own regions. Requery after cloning, replacing or moving the unit.
bool isDefinedInside(mlir::Operation *operation, mlir::Value value);
llvm::SmallVector<mlir::Value> externalOperands(mlir::Operation *operation);

// A flat read epoch permits reordering ordinary reads, not writes, allocation,
// synchronization, volatile access or an operation with unknown effects.
bool isReadEpoch(mlir::Block &block,
                 llvm::function_ref<bool(mlir::Operation *)> ordinaryRead);

// Return a dependency-ordered, bounded slice that can move before the anchor.
// The caller supplies the target/type restriction; dominance, speculation and
// effect legality are shared. No source read is included in this pure slice.
mlir::FailureOr<llvm::SmallVector<mlir::Operation *>> movablePureSlice(
    mlir::ValueRange values, mlir::Operation *anchor,
    llvm::function_ref<bool(mlir::Operation *)> accepts, unsigned limit);

struct DependencyCost {
  double criticalPath = 0;
  double carriedPath = 0;
};
// Costs describe the current SSA block and are never cached across rewrites.
// carriedPath is an upper bound over all carry-to-yield paths, including paths
// between different carries. The target owns instruction latency estimates.
DependencyCost dependencyCost(mlir::Block &block, mlir::ValueRange carries,
                              llvm::function_ref<double(mlir::Operation *)> latency);

// Exact cardinality when SCF bounds are constant, including empty loops and
// signed/unsigned comparison. Unknown bounds never receive a guessed count.
std::optional<uint64_t> staticTripCount(mlir::Operation *operation);
struct ExecutionMultiplicity {
  uint64_t fixed = 1;
  llvm::SmallVector<mlir::Operation *> dynamicLoops;
  // When requested, retain the actual branch domains instead of assigning a
  // probability. All surrounding loops then stay symbolic, including static
  // loops: the predicate may depend on their induction values.
  llvm::SmallVector<std::pair<mlir::Value, bool>> conditions;
};
// Frequency relative to one execution of the block containing scope. Unknown
// loop counts remain independent symbols; conditional/opaque regions are not
// assigned fictitious probabilities. The query owns no persistent IR facts.
std::optional<ExecutionMultiplicity> executionMultiplicity(
    mlir::Operation *operation, mlir::Operation *scope);
// Frequency per execution of one enclosing block after target control lowering.
// Unknown counts remain SSA symbols. By default intervening non-loop regions
// make the query unknown. preserveConditionalDomain also records SCF if domains
// symbolically; its callers must include those domains when comparing costs.
std::optional<ExecutionMultiplicity> executionMultiplicityInBlock(
    mlir::Operation *operation, mlir::Block *scope,
    bool preserveConditionalDomain = false);
} // namespace weft::analysis
#endif
