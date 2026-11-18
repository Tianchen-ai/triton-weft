#pragma once

#include "Weft/Target/X86/X86Target.h"
#include "mlir/IR/Builders.h"
#include <optional>

namespace weft::x86 {
// Map a row-major [M,K] × [K,N] region to register-resident FP32 rows.
std::optional<std::pair<mlir::Value, mlir::Value>> emitDenseOuterProduct(
    mlir::OpBuilder &b, mlir::Operation &op, const X86Target &target,
    mlir::ValueRange inputs, mlir::ValueRange masks);
}
