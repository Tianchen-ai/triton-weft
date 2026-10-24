#ifndef WEFT_ANALYSIS_INTEGER_FACTS_H
#define WEFT_ANALYSIS_INTEGER_FACTS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <optional>

namespace weft::analysis {

// Mathematical bounds for every lane, independent of validity and storage.
// Unknown/wrapping results retain their entire type domain. The current exact
// interval representation covers integer types through 62 bits; wider types
// return no proof, never a truncated bound.
struct IntegerRange {
  int64_t minimum;
  int64_t maximum;
  bool fits(mlir::IntegerType type) const;
};

std::optional<IntegerRange> integerTypeRange(mlir::IntegerType type);
std::optional<IntegerRange> productSumRange(IntegerRange lhs, IntegerRange rhs,
                                           int64_t group);
bool saturatingProductSumIsExact(IntegerRange lhs, IntegerRange rhs,
                                 int64_t group, mlir::IntegerType accumulator);

// Targets describe their own operations; the shared interpreter owns interval,
// conversion and overflow reasoning. It does not select instructions or layouts.
enum class IntegerExpressionKind {
  Unknown, Known, Preserve, ZeroOr, Cast, Saturate, Union, Add, Sub, Mul, And, Or,
  ShiftLeft, ShiftRight, Divide, ProductSum, Sum
};

struct IntegerExpression {
  mlir::IntegerType type;
  IntegerExpressionKind kind = IntegerExpressionKind::Unknown;
  llvm::SmallVector<mlir::Value, 3> inputs;
  llvm::SmallVector<mlir::IntegerType, 3> interpretations;
  std::optional<IntegerRange> known;
  int64_t group = 1;
};

using IntegerDescription =
    std::function<std::optional<IntegerExpression>(mlir::Value)>;

IntegerExpressionKind integerBinaryKind(llvm::StringRef name);
mlir::IntegerType integerElementType(mlir::Type type);

// A query has a finite traversal budget and no persistent facts to invalidate.
// Signless machine carriers require an explicit signed/unsigned interpretation.
// Unknown operations, cycles and unsupported widths cannot establish a proof.
std::optional<IntegerRange> integerRange(
    mlir::Value value, mlir::IntegerType interpretation = {},
    const IntegerDescription &targetDescription = {});

} // namespace weft::analysis
#endif
