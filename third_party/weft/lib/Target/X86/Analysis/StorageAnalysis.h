#ifndef WEFT_X86_STORAGE_ANALYSIS_H
#define WEFT_X86_STORAGE_ANALYSIS_H

#include "Weft/Analysis/StorageLifetime.h"

namespace weft::x86 {
// Compiler policy for bounded private residency, not a hardware capacity.
inline constexpr uint64_t privateStorageBudget = 64 * 1024;
// The alias closure must fit the allocation's dynamic lifetime. In particular,
// a buffer born in a loop cannot be promoted if a back edge retains its address.
mlir::FailureOr<analysis::StorageUses> privateStorageUses(
    mlir::Value allocation, mlir::Operation *iteration = nullptr);
// All possible origins must be fresh, nonescaping allocations. This is only an
// alias/effect proof, not a coordinate mapping for a view or tile transfer.
bool isPrivateStorage(mlir::Value value);
}
#endif
