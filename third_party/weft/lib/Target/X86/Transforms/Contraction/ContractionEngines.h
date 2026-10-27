#ifndef WEFT_X86_CONTRACTION_ENGINES_H
#define WEFT_X86_CONTRACTION_ENGINES_H

#include "Weft/Target/X86/X86Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace weft::x86 {
enum class ContractionEngine { Vector, MatrixInteger, MatrixBFloat };

struct ContractionStorage {
  bool leftArray = false;
  bool rightArray = false;
  bool rightVector = false;
};

bool matrixReductionFirst(mlir::Operation &op, unsigned operand);
llvm::SmallVector<int64_t> matrixOperandShape(mlir::Operation &op, unsigned operand,
                                            llvm::ArrayRef<int64_t> shape);

bool integerMatrixGeometry(mlir::Operation &op, const X86Target &target,
                           llvm::ArrayRef<int64_t> left, llvm::ArrayRef<int64_t> right);
mlir::FailureOr<ContractionEngine> selectContractionEngine(
    mlir::Operation &op, const X86Target &target, ContractionStorage storage,
    bool arrayResult, llvm::ArrayRef<int64_t> left, llvm::ArrayRef<int64_t> right,
    llvm::ArrayRef<int64_t> output);
void recordContractionEngine(mlir::ModuleOp module, mlir::Operation &source,
                             ContractionEngine engine);
} // namespace weft::x86
#endif
