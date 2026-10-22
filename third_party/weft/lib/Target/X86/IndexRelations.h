#pragma once

#include "mlir/IR/Value.h"

namespace weft::x86 {
// Prove consecutive, non-wrapping integer coordinates along a logical axis.
// This describes the index expression, not a View's physical strides or mask.
bool hasNonWrappingUnitStride(mlir::Value value, int64_t axis);
bool hasUnsignedBound(mlir::Value value, uint64_t exclusiveUpper);
} // namespace weft::x86
