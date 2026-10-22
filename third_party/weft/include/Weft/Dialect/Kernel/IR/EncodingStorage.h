#ifndef WEFT_KERNEL_ENCODING_STORAGE_H
#define WEFT_KERNEL_ENCODING_STORAGE_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace weft::kernel {

// Caller-visible storage is fixed by the author declaration, before any target
// or instruction selection. Both interface analysis and physical lowering use
// this same resolution of a base record and its explicit interleave program.
struct EncodingStorage {
  EncodingDeclOp base;
  int64_t rows = 1;
  llvm::SmallVector<FieldInterleaveSpan> spans;

  int64_t bytes() { return base.getStorageBits() / 8; }
  int64_t elements() { return base.getElements(); }
};

inline mlir::FailureOr<EncodingStorage> encodingStorage(mlir::ModuleOp module,
                                                       EncodingType encoding) {
  EncodingStorage result;
  llvm::StringRef family = encoding.getFamily();
  InterleaveOp interleave;
  if (encoding.getKind() == "derived_instance") {
    for (auto derive : module.getOps<DeriveOp>()) {
      if (derive.getResultFamily() != family || derive.getLayoutIdentity() != encoding.getLayoutIdentity() ||
          derive.getParameterValues() != encoding.getParameters().asArrayRef()) continue;
      auto &body = derive.getBody().front();
      if (body.getOperations().size() != 2 || !(interleave = mlir::dyn_cast<InterleaveOp>(body.front())))
        return derive.emitError("artifact storage requires the declared interleave program");
      family = derive.getSourceFamily();
      result.rows = interleave.getRows();
      break;
    }
    if (!interleave) return mlir::failure();
  }
  for (auto declaration : module.getOps<EncodingDeclOp>())
    if (declaration.getSymName() == family) result.base = declaration;
  if (!result.base || result.base.getStorageBits() % 8) return mlir::failure();
  if (interleave) {
    auto spans = fieldInterleaveSpans(result.base.getStorageBits(), result.base.getFieldBitOffsets(),
        result.base.getFieldStorageBits(), result.base.getPadding(), interleave.getFieldChunks());
    if (mlir::failed(spans)) return mlir::failure();
    result.spans = std::move(*spans);
  }
  return result;
}
} // namespace weft::kernel
#endif
