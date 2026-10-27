#ifndef WEFT_X86_CONTRACTION_OUTPUTS_H
#define WEFT_X86_CONTRACTION_OUTPUTS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace weft::x86 {
struct OutputContraction {
  mlir::scf::ForOp outer, reduction;
};
// Packetize independent outputs of a closed array contraction. The numerical
// reduction remains inside each lane; invariant operand coordinates share one
// scalar supply. The caller owns cost/resource selection and the scalar tail.
mlir::FailureOr<OutputContraction> packetizeContractionOutputs(
    mlir::IRRewriter &b, mlir::scf::ForOp source, unsigned lanes);
} // namespace weft::x86
#endif
