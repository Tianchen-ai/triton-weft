#ifndef WEFT_TARGET_FIELDREADS_H
#define WEFT_TARGET_FIELDREADS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace weft::riscv_internal {
mlir::LogicalResult materializeFieldReads(mlir::ModuleOp module,
                                         mlir::IRRewriter &rewriter);
}

#endif
