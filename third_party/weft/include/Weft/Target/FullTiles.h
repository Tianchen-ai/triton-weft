#ifndef WEFT_TARGET_FULL_TILES_H
#define WEFT_TARGET_FULL_TILES_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace weft {
struct FullTileGrowth {
  int64_t budget = 0;
  int64_t operations = 0;
  int64_t domainVersions = 0;
};

// Partition existing SCF iterations, retaining their original carries, body
// effects and masked remainder. Unsigned coordinates retain their full range.
mlir::FailureOr<FullTileGrowth> separateFullTiles(mlir::Operation *function,
                                                bool unsignedIndices);
} // namespace weft
#endif
