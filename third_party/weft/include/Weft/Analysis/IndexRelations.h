#pragma once

#include "mlir/IR/Value.h"

namespace weft::analysis {
// Prove consecutive, non-wrapping integer coordinates along a logical axis.
// This describes the index expression, not a View's physical strides or mask.
bool hasNonWrappingUnitStride(mlir::Value value, int64_t axis);
bool hasUnsignedBound(mlir::Value value, uint64_t exclusiveUpper);
// Every logical position maps to its row-major ordinal, without wrap. This is
// an index-expression fact, independent of the payload's physical layout.
bool hasRowMajorCoordinates(mlir::Value value);
} // namespace weft::analysis
