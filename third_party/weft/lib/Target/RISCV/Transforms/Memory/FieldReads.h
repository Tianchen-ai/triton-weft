#ifndef WEFT_TARGET_FIELDREADS_H
#define WEFT_TARGET_FIELDREADS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace weft::riscv_internal {
mlir::LogicalResult materializeFieldReads(mlir::ModuleOp module,
    mlir::IRRewriter &rewriter,
    llvm::function_ref<riscv::AccessAttr(riscv::FieldOp, mlir::Type)> selectAccess);
}

#endif
