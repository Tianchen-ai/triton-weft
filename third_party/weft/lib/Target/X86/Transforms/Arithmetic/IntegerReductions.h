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

// Four adjacent products in a machine i32 carrier. Numerical proofs choose
// VNNI bytes or exact AVX2 pairs; callers restore a narrower source boundary.
std::optional<IntegerPairResult> selectX86IntegerQuads(
    mlir::ModuleOp module, mlir::OpBuilder &builder, mlir::Location location,
    mlir::Value lhs, mlir::Value rhs, mlir::Value lhsValid, mlir::Value rhsValid,
    mlir::IntegerType lhsElement, mlir::IntegerType rhsElement,
    const X86Target &target);

// Adjacent even groups retain their source boundary while composing machine
// pair/quad products with an integer tree and the same grouped validity.
std::optional<IntegerPairResult> selectX86IntegerGroups(
    mlir::ModuleOp module, mlir::OpBuilder &builder, mlir::Location location,
    mlir::Value lhs, mlir::Value rhs, mlir::Value lhsValid, mlir::Value rhsValid,
    mlir::IntegerType lhsElement, mlir::IntegerType rhsElement,
    int64_t group, unsigned resultBits, const X86Target &target);
} // namespace weft
#endif
