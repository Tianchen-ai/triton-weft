#ifndef WEFT_TARGET_X86_ARRAYS_H
#define WEFT_TARGET_X86_ARRAYS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <functional>

namespace weft::x86 {
struct ArrayStorage {
  mlir::Value data, valid;
  llvm::SmallVector<mlir::Value> extents;
};
inline mlir::Value arraySize(mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange extents) {
  mlir::Value count = mlir::arith::ConstantIndexOp::create(b, loc, 1);
  for (mlir::Value extent : extents) count = mlir::arith::MulIOp::create(b, loc, count, extent);
  return count;
}
inline ArrayStorage allocateArray(mlir::OpBuilder &b, mlir::Location loc,
                                  mlir::Type element, mlir::ValueRange extents) {
  mlir::Value count = arraySize(b, loc, extents);
  auto type = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, element);
  auto mask = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, b.getI8Type());
  auto data = mlir::memref::AllocOp::create(b, loc, type, count);
  // Private storage serves complete SIMD packets and AMX cache-line rows.
  // This aligns the owned buffer; it grants no alignment to caller memory.
  data->setAttr("alignment", b.getI64IntegerAttr(64));
  auto valid = mlir::memref::AllocOp::create(b, loc, mask, count);
  valid->setAttr("weft.x86.validity_array", b.getUnitAttr());
  return {data, valid, llvm::to_vector(extents)};
}
inline mlir::LogicalResult forArray(mlir::OpBuilder &b, mlir::Location loc,
    mlir::ValueRange extents,
    llvm::function_ref<mlir::LogicalResult(mlir::Value, mlir::ValueRange)> body,
    bool elementwise = false, mlir::Attribute sourceScope = {}) {
  auto zero = mlir::arith::ConstantIndexOp::create(b, loc, 0);
  auto one = mlir::arith::ConstantIndexOp::create(b, loc, 1);
  auto loop = mlir::scf::ForOp::create(b, loc, zero, arraySize(b, loc, extents), one);
  if (elementwise) loop->setAttr("weft.x86.array_elementwise", b.getUnitAttr());
  if (sourceScope) loop->setAttr("weft.x86.array_scope", sourceScope);
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(loop.getBody());
  mlir::Value linear = loop.getInductionVar(), remaining = linear;
  llvm::SmallVector<mlir::Value> coordinates(extents.size());
  for (int64_t axis = extents.size() - 1; axis >= 0; --axis) {
    // The traversal is bounded by the product of these extents: after removing
    // trailing axes, the leading coordinate is already in range.
    coordinates[axis] = axis == 0 ? remaining :
        mlir::Value(mlir::arith::RemUIOp::create(b, loc, remaining, extents[axis]));
    if (axis) remaining = mlir::arith::DivUIOp::create(b, loc, remaining, extents[axis]);
  }
  return body(linear, coordinates);
}
// Pack only within a logical row. A leading-axis gap never becomes an implied
// contiguous memory access, and the scalar tail stays inside the source domain.
inline mlir::LogicalResult forArrayPackets(mlir::OpBuilder &b, mlir::Location loc,
    mlir::ValueRange extents, int64_t lanes,
    llvm::function_ref<mlir::LogicalResult(mlir::Value, mlir::ValueRange, int64_t)> body) {
  return forArray(b, loc, extents.drop_back(), [&](mlir::Value row, mlir::ValueRange outer) {
    auto zero = mlir::arith::ConstantIndexOp::create(b, loc, 0);
    auto one = mlir::arith::ConstantIndexOp::create(b, loc, 1);
    auto width = mlir::arith::ConstantIndexOp::create(b, loc, lanes);
    mlir::Value count = extents.back();
    mlir::Value base = mlir::arith::MulIOp::create(b, loc, row, count);
    mlir::Value end = mlir::arith::SubIOp::create(b, loc, count,
        mlir::arith::RemUIOp::create(b, loc, count, width));
    auto packet = mlir::scf::ForOp::create(b, loc, zero, end, width);
    packet->setAttr("weft.x86.memory_packet_lanes", b.getI64IntegerAttr(lanes));
    {
      mlir::OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(packet.getBody());
      auto type = mlir::VectorType::get({lanes}, b.getIndexType());
      llvm::SmallVector<mlir::Value> coordinates;
      for (auto coordinate : outer)
        coordinates.push_back(mlir::vector::BroadcastOp::create(b, loc, type, coordinate));
      llvm::SmallVector<int64_t> sequence;
      for (int64_t lane = 0; lane < lanes; ++lane) sequence.push_back(lane);
      mlir::Value offsets = mlir::arith::ConstantOp::create(b, loc, type,
          mlir::DenseIntElementsAttr::get(type, sequence));
      coordinates.push_back(mlir::arith::AddIOp::create(b, loc,
          mlir::vector::BroadcastOp::create(b, loc, type, packet.getInductionVar()), offsets));
      auto linear = mlir::arith::AddIOp::create(b, loc, base, packet.getInductionVar());
      if (failed(body(linear, coordinates, lanes))) return mlir::failure();
    }
    auto tail = mlir::scf::ForOp::create(b, loc, end, count, one);
    mlir::OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(tail.getBody());
    llvm::SmallVector<mlir::Value> coordinates(outer); coordinates.push_back(tail.getInductionVar());
    return body(mlir::arith::AddIOp::create(b, loc, base, tail.getInductionVar()), coordinates, 1);
  });
}
inline mlir::Value arrayLinear(mlir::OpBuilder &b, mlir::Location loc,
                               mlir::ValueRange extents, mlir::ValueRange coordinates) {
  mlir::Value index = mlir::arith::ConstantIndexOp::create(b, loc, 0);
  for (auto [extent, coordinate] : llvm::zip(extents, coordinates))
    index = mlir::arith::AddIOp::create(b, loc, mlir::arith::MulIOp::create(b, loc, index, extent), coordinate);
  return index;
}
} // namespace weft::x86
#endif
