#ifndef WEFT_X86_INTEGER_REDUCTIONS_H
#define WEFT_X86_INTEGER_REDUCTIONS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <optional>

namespace weft {
struct X86Target;
struct IntegerPairResult {
  mlir::Value value;
  mlir::Value valid;
};

// Operands have the same flattened domain with adjacent logical pairs.
// Signedness comes from the Canonical element types, not the signless carriers.
std::optional<IntegerPairResult> selectX86IntegerPairs(
    mlir::ModuleOp module, mlir::OpBuilder &builder, mlir::Location location,
    mlir::Value lhs, mlir::Value rhs, mlir::Value lhsValid, mlir::Value rhsValid,
    mlir::IntegerType lhsElement, mlir::IntegerType rhsElement,
    unsigned resultBits, unsigned vectorBits);

// Four adjacent products, with exact modular i32 accumulation. Signed bytes
// use an explicit unsigned bias and correction; no saturating intermediate.
std::optional<IntegerPairResult> selectX86ByteDotGroups(
    mlir::ModuleOp module, mlir::OpBuilder &builder, mlir::Location location,
    mlir::Value lhs, mlir::Value rhs, mlir::Value lhsValid, mlir::Value rhsValid,
    mlir::IntegerType lhsElement, mlir::IntegerType rhsElement,
    const X86Target &target);
} // namespace weft
#endif
