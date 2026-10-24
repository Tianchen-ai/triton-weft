#ifndef WEFT_TARGET_RISCV_NUMERICCONVERSIONS_H
#define WEFT_TARGET_RISCV_NUMERICCONVERSIONS_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
namespace weft::riscv_internal {
mlir::LogicalResult materializeNumericConversions(mlir::ModuleOp module, mlir::IRRewriter &rewriter);
}
#endif
