#pragma once

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace weft::kernel {
// Logical source domains, independent of target storage and physical lanes.
mlir::Type logicalElement(mlir::Type type);
llvm::ArrayRef<int64_t> logicalShape(mlir::Type type);
llvm::ArrayRef<int64_t> logicalAxes(mlir::Type type);
bool sameDomain(mlir::Type lhs, mlir::Type rhs);
std::optional<int64_t> staticElementCount(mlir::Type type);
} // namespace weft::kernel
