#ifndef WEFT_LIB_TARGET_RISCVRESIDENCYPLANNING_H
#define WEFT_LIB_TARGET_RISCVRESIDENCYPLANNING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir { class IRRewriter; }

namespace weft {
namespace riscv { class MaterializeOp; }

mlir::LogicalResult planRISCVResidency(mlir::ModuleOp module);
// The candidate evaluator and the main lowering materialize the same physical
// supply graph. Local allocations/packs have their own lowering owner.
mlir::LogicalResult materializeRISCVSharedSupply(riscv::MaterializeOp operation,
                                                mlir::IRRewriter &rewriter);

} // namespace weft

#endif
