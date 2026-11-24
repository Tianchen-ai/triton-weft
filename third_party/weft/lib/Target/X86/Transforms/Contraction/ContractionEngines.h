#ifndef WEFT_X86_CONTRACTION_ENGINES_H
#define WEFT_X86_CONTRACTION_ENGINES_H

#include "Weft/Target/X86/X86Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace weft::x86 {
enum class ContractionEngine {
  Vector,
  MatrixInteger,
  MatrixBFloat,
  OneDNNBrgemmF32,
  OneDNNBrgemmF16,
  OneDNNBrgemmBF16,
  OneDNNBrgemmS8,
  OneDNNBrgemmU8S8
};

inline bool isOneDNNBrgemm(ContractionEngine engine) {
  return engine == ContractionEngine::OneDNNBrgemmF32 ||
         engine == ContractionEngine::OneDNNBrgemmF16 ||
         engine == ContractionEngine::OneDNNBrgemmBF16 ||
         engine == ContractionEngine::OneDNNBrgemmS8 ||
         engine == ContractionEngine::OneDNNBrgemmU8S8;
}

struct ContractionStorage {
  bool leftArray = false;
  bool rightArray = false;
  bool rightVector = false;
  bool leftVector = false;
};

struct BrgemmGeometry {
  bool valid = false;
  bool packed3D = false;
  bool vectorRight = false;
  bool leftReductionFirst = false;
  bool rightReductionFirst = false;
  int64_t staticM = -1;
  int64_t staticN = -1;
  int64_t staticK = -1;
  int64_t kBlock = 1;
};

bool matrixReductionFirst(mlir::Operation &op, unsigned operand);
llvm::SmallVector<int64_t> matrixOperandShape(mlir::Operation &op, unsigned operand,
                                              llvm::ArrayRef<int64_t> shape);
BrgemmGeometry describeBrgemmGeometry(mlir::Operation &op,
                                      llvm::ArrayRef<int64_t> left,
                                      llvm::ArrayRef<int64_t> right,
                                      llvm::ArrayRef<int64_t> output);

bool integerMatrixGeometry(mlir::Operation &op, const X86Target &target,
                           llvm::ArrayRef<int64_t> left, llvm::ArrayRef<int64_t> right);
mlir::FailureOr<ContractionEngine> selectContractionEngine(
    mlir::Operation &op, const X86Target &target, llvm::StringRef contractionBackend,
    ContractionStorage storage, bool arrayResult,
    llvm::ArrayRef<int64_t> left, llvm::ArrayRef<int64_t> right,
    llvm::ArrayRef<int64_t> output, bool emitDiagnostics = true);
void recordContractionEngine(mlir::ModuleOp module, mlir::Operation &source,
                             ContractionEngine engine);
} // namespace weft::x86
#endif
