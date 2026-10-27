#ifndef WEFT_X86_ARRAY_PACKETS_H
#define WEFT_X86_ARRAY_PACKETS_H
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace weft::x86 {
// The input is an owned scalar array. Only the physical reduction partials
// change; validity, element type and the original reduction boundary are kept.
std::pair<mlir::Value, mlir::Value> reduceArrayPackets(
    mlir::OpBuilder &b, mlir::Location loc, mlir::Value data, mlir::Value valid,
    mlir::Value base, mlir::Value stride, mlir::Value count, mlir::Value identity,
    mlir::vector::CombiningKind kind, llvm::StringRef scalarOperation, unsigned vectorBits,
    mlir::Attribute sourceScope);
}
#endif
