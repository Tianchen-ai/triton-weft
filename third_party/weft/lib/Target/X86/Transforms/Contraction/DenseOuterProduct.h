#pragma once

#include "Weft/Target/X86/X86Target.h"
#include "mlir/IR/Builders.h"
#include <optional>

namespace weft::x86 {
// Accumulate [M,K] against [K,N] or [N,K] in register-sized row groups.
std::optional<std::pair<mlir::Value, mlir::Value>> emitDenseOuterProduct(
    mlir::OpBuilder &b, mlir::Operation &op, const X86Target &target,
    mlir::ValueRange inputs, mlir::ValueRange masks,
    mlir::Value initAcc = {}, mlir::Value initAccMask = {});
}
