#ifndef WEFT_X86_ONEDNN_CONTRACTIONS_H
#define WEFT_X86_ONEDNN_CONTRACTIONS_H

#include "Transforms/Contraction/ContractionEngines.h"
#include "Transforms/Memory/Arrays.h"
#include "Weft/Target/X86/X86Target.h"
#include "mlir/IR/BuiltinOps.h"
#include <optional>

namespace weft::x86 {

struct DirectContractionView {
  mlir::Value base;    // !llvm.ptr
  mlir::Value offset;  // index (in logical elements)
  mlir::Value guard;   // i1
  llvm::SmallVector<mlir::Value> extents;
  llvm::SmallVector<mlir::Value> strides;
  llvm::SmallVector<mlir::Value> limits;
  mlir::Type elementType;
};

struct BrgemmOperand {
  std::optional<DirectContractionView> view;
  ArrayStorage array;
  mlir::Type logicalElement;
  mlir::Value laneValidity{};
};

mlir::LogicalResult materializeOneDNNBrgemm(
    mlir::ModuleOp module, mlir::OpBuilder &b, mlir::Location loc,
    const X86Target &target, ContractionEngine engine,
    const BrgemmGeometry &geom, const BrgemmOperand &lhs,
    const BrgemmOperand &rhs, const ArrayStorage &output, bool accumulate);

// Fast native SIMD outer-product / inner-product lowering for dynamic or
// static 2D/3D array contractions when ContractionEngine::Vector is selected.
bool canMaterializeSIMDArrayContraction(
    const X86Target &target, const BrgemmGeometry &geom,
    mlir::Type lhsElem, mlir::Type rhsElem, mlir::Type resElem);
bool materializeSIMDArrayContraction(
    mlir::OpBuilder &b, mlir::Location loc, const X86Target &target,
    const BrgemmGeometry &geom, const ArrayStorage &lhs, mlir::Type lhsElem,
    const ArrayStorage &rhs, mlir::Type rhsElem,
    const ArrayStorage &output, mlir::Type resElem);

} // namespace weft::x86

#endif
