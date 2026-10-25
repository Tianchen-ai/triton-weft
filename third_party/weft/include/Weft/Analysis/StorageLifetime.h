#ifndef WEFT_ANALYSIS_STORAGE_LIFETIME_H
#define WEFT_ANALYSIS_STORAGE_LIFETIME_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace weft::analysis {
// Structured SSA transports addresses without creating a new storage object.
// Include initial and back edges; callers close cycles with their visited set.
// Target-specific views, allocations and snapshot boundaries stay in the target.
bool incomingControlValues(mlir::Value value,
                           llvm::SmallVectorImpl<mlir::Value> &incoming);
bool outgoingControlAliases(mlir::OpOperand &use,
                            llvm::SmallVectorImpl<mlir::Value> &aliases);

struct StorageUses {
  llvm::SmallVector<mlir::Value> aliases;
  llvm::SmallVector<mlir::Operation *> users;
};
// Check every operand, including repeated operands of the same call. `follow`
// must either prove a non-retaining use or append every possible outgoing alias.
// `contains` supplies the owner's lifetime boundary, including iteration scope.
// Results are local to the current IR; rewrites require a fresh query.
mlir::FailureOr<StorageUses> collectStorageUses(
    mlir::Value root, llvm::function_ref<bool(mlir::Value)> contains,
    llvm::function_ref<bool(mlir::OpOperand &,
                            llvm::SmallVectorImpl<mlir::Value> &)> follow);

enum class StorageControl { Opaque, Branch, Repeated };
StorageControl structuredStorageControl(mlir::Operation *op);
// An unchanged read epoch from a completed read to a later consumer. A loop
// containing the consumer requires its complete body to preserve the storage:
// a write after the consumer affects the next iteration. A branch needs only
// the prefix actually containing the consumer. Unknown control stops the proof.
bool storagePreservedBetween(
    mlir::Operation *read, mlir::Operation *consumer,
    llvm::function_ref<bool(mlir::Operation *)> preserves,
    llvm::function_ref<StorageControl(mlir::Operation *)> control =
        structuredStorageControl);
} // namespace weft::analysis
#endif
