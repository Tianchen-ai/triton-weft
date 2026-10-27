#ifndef WEFT_TRANSFORMS_FULL_TILES_H
#define WEFT_TRANSFORMS_FULL_TILES_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace weft {
struct FullTileGrowth {
  int64_t budget = 0;
  int64_t operations = 0;
  int64_t domainVersions = 0;
  llvm::SmallVector<mlir::Operation *> fullDomains;
};

// Partition existing SCF iterations, retaining their original carries, body
// effects and masked remainder. Unsigned coordinates retain their full range.
mlir::FailureOr<FullTileGrowth> separateFullTiles(mlir::Operation *function,
    bool unsignedIndices,
    llvm::function_ref<int64_t(mlir::Operation *)> guardedPayloads = {},
    llvm::function_ref<bool(mlir::Operation *)> jointGuardedDomains = {});
} // namespace weft
#endif
