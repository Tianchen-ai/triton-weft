#ifndef WEFT_X86_AMX_CONTRACTIONS_H
#define WEFT_X86_AMX_CONTRACTIONS_H

#include "Transforms/Memory/Arrays.h"
#include "Weft/Target/X86/X86Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace weft::x86 {
// Complete BF16 snapshots and a closed F32 contraction. The guard
// preserves validity and the normal product lattice before AMX execution.
mlir::Value guardAMXBFloat(mlir::OpBuilder &b, mlir::Location loc,
                           const ArrayStorage &input, int64_t count, int64_t terms);
void materializeAMXBFloat(mlir::ModuleOp module, mlir::OpBuilder &b,
                          mlir::Location loc, const X86Target &target,
                          const ArrayStorage &left, const ArrayStorage &right,
                          const ArrayStorage &output,
                          int64_t rows, int64_t columns, int64_t terms,
                          bool leftReductionFirst = false, bool rightReductionFirst = false);
// Byte snapshots. Signedness selects the exact modular I32 dot. The masked
// plan zeroes inactive saved lanes; its caller materializes pairwise validity.
void materializeAMXInteger(mlir::ModuleOp module, mlir::OpBuilder &b,
                          mlir::Location loc, const X86Target &target,
                          const ArrayStorage &left, const ArrayStorage &right,
                          const ArrayStorage &output,
                          int64_t rows, int64_t columns, int64_t terms,
                          bool leftUnsigned, bool rightUnsigned,
                          bool leftReductionFirst = false, bool rightReductionFirst = false,
                          bool maskInputs = false);
// Return the row-major SSA result of the closed fixed contraction. Tiled output
// uses the private destination as tile-major scratch; array consumers must keep
// row-major storage.
mlir::Value materializeAMXIntegerFixed(mlir::ModuleOp module, mlir::OpBuilder &b,
                               mlir::Location loc, const X86Target &target,
                               const ArrayStorage &left, mlir::Value right,
                               const ArrayStorage &output,
                               int64_t rows, int64_t columns, int64_t terms,
                               bool leftUnsigned, bool rightUnsigned,
                               bool tiledOutput = false, bool leftReductionFirst = false,
                               bool maskInputs = false);
std::unique_ptr<mlir::Pass> createAMXConfigurationReusePass();
std::unique_ptr<mlir::Pass> createAMXSnapshotTransferPass();
std::unique_ptr<mlir::Pass> createAMXPackingReusePass();
std::unique_ptr<mlir::Pass> createAMXLoadSchedulePass();
}
#endif
