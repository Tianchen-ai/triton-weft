#ifndef WEFT_RISCV_NUMERICAL_VALIDITY_H
#define WEFT_RISCV_NUMERICAL_VALIDITY_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

namespace weft::riscv_internal {
mlir::Value activeNumericSubrange(mlir::IRRewriter &b, mlir::Location loc,
                                 mlir::Value bound, mlir::Value offset,
                                 int64_t capacity);
// Active rectangular extents of a saved numeric value. These are logical
// coordinates, not register capacity or permission to repeat a source read.
mlir::FailureOr<llvm::SmallVector<mlir::Value>> activeNumericExtents(
    mlir::IRRewriter &b, mlir::Value value);

// Materialize validity along the original use-def graph before replacing a
// closed primitive. Effects consume explicit predicates in the physical IR.
mlir::LogicalResult propagateNumericValidity(mlir::IRRewriter &b,
    llvm::ArrayRef<mlir::Operation *> operations,
    llvm::DenseMap<mlir::Value, mlir::Value> &masks, int64_t &nextAxis);
}
#endif
