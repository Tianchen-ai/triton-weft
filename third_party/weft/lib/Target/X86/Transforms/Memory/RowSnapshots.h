#ifndef WEFT_X86_ROW_SNAPSHOTS_H
#define WEFT_X86_ROW_SNAPSHOTS_H

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace weft::kernel { class UpdateOp; }
namespace weft::x86 {
// Capture an entire rectangular read at its source position. The callback
// transfers one packet using the original memory encoding and stride.
std::pair<mlir::Value, mlir::Value> loadRowSnapshot(
    mlir::OpBuilder &b, mlir::Location loc, mlir::VectorType flat, int64_t columns,
    mlir::Value offset, mlir::Value guard, mlir::ValueRange strides, mlir::ValueRange limits,
    llvm::function_ref<mlir::Value(mlir::Value, mlir::Value)> loadPacket);

std::optional<std::pair<mlir::Value, mlir::Value>> updateRowWindow(
    mlir::OpBuilder &b, weft::kernel::UpdateOp update, mlir::Value indices, mlir::Value indexMask,
    mlir::Value owner, mlir::Value ownerMask, mlir::Value values, mlir::Value valid);
}
#endif
