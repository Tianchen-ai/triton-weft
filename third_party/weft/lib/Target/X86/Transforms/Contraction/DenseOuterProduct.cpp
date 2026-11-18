#include "Transforms/Contraction/DenseOuterProduct.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/TypeQueries.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace wk = weft::kernel;

std::optional<std::pair<Value, Value>> weft::x86::emitDenseOuterProduct(
    OpBuilder &b, Operation &op, const X86Target &target, ValueRange inputs, ValueRange masks) {
  if (!isa<wk::OuterContractOp>(op) || !wk::logicalElement(op.getResult(0).getType()).isF32()) return {};
  auto left = op.getOperand(0).getType(), right = op.getOperand(1).getType();
  auto la = wk::logicalAxes(left), ra = wk::logicalAxes(right);
  auto over = op.getAttrOfType<DenseI64ArrayAttr>("over").asArrayRef();
  if (la.size() != 2 || ra.size() != 2 || over.size() != 1 ||
      la[1] != over[0] || ra[0] != over[0] || la[0] == ra[1]) return {};
  for (Value input : inputs) {
    auto v = dyn_cast<VectorType>(input.getType());
    if (!v || !(v.getElementType().isF32() || v.getElementType().isF16() || v.getElementType().isBF16()))
      return {};
  }
  int64_t rows = wk::logicalShape(left)[0], depth = wk::logicalShape(left)[1];
  int64_t cols = wk::logicalShape(right)[1], lanes = target.vectorBits / 32;
  if (rows <= 0 || cols < lanes || depth < 2 * lanes ||
      (2 * rows + 4) * llvm::divideCeil(cols, lanes) > target.vectorRegisters()) return {};

  Location loc = op.getLoc();
  auto index = [&](int64_t v) { return arith::ConstantIndexOp::create(b, loc, v).getResult(); };
  auto zero = [&](Type t) { return arith::ConstantOp::create(b, loc, t, b.getZeroAttr(t)).getResult(); };
  Value start = index(0);
  auto snapshot = [&](Value v) {
    auto type = cast<VectorType>(v.getType());
    // The existing private-storage pass assigns and reuses bounded stack slots.
    Value buffer = memref::AllocOp::create(b, loc, MemRefType::get({type.getNumElements()}, type.getElementType()));
    vector::StoreOp::create(b, loc, v, buffer, start);
    return buffer;
  };
  auto maskSnapshot = [&](Value mask) -> Value {
    if (matchPattern(mask, m_One())) return {};
    auto bytes = cast<VectorType>(mask.getType()).clone(b.getI8Type());
    return snapshot(arith::ExtUIOp::create(b, loc, bytes, mask));
  };
  Value a = snapshot(inputs[0]), weights = snapshot(inputs[1]);
  Value av = maskSnapshot(masks[0]), bv = maskSnapshot(masks[1]);
  auto rowType = VectorType::get({cols}, b.getF32Type());
  auto maskType = rowType.clone(b.getI1Type());
  Value noValues = zero(rowType), noLanes = zero(maskType);
  Value allLanes = arith::ConstantOp::create(b, loc, maskType, DenseElementsAttr::get(maskType, true));
  SmallVector<Value> initial(rows, noValues);
  initial.append(rows, noLanes);
  auto loop = scf::ForOp::create(b, loc, start, index(depth), index(1), initial);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    Value base = arith::MulIOp::create(b, loc, k, index(cols));
    auto rhsType = rowType.clone(cast<MemRefType>(weights.getType()).getElementType());
    Value rhs = vector::LoadOp::create(b, loc, rhsType, weights, base);
    if (rhsType != rowType) rhs = arith::ExtFOp::create(b, loc, rowType, rhs);
    Value rightLive = allLanes;
    if (bv) rightLive = arith::TruncIOp::create(b, loc, maskType,
        vector::LoadOp::create(b, loc, rowType.clone(b.getI8Type()), bv, base));
    SmallVector<Value> sums, validity;
    for (int64_t m = 0; m < rows; ++m) {
      Value offset = arith::AddIOp::create(b, loc, index(m * depth), k);
      Value lhs = memref::LoadOp::create(b, loc, a, offset);
      if (!lhs.getType().isF32()) lhs = arith::ExtFOp::create(b, loc, b.getF32Type(), lhs);
      lhs = vector::BroadcastOp::create(b, loc, rowType, lhs);
      Value live = rightLive;
      if (av) {
        Value valid = arith::TruncIOp::create(b, loc, b.getI1Type(), memref::LoadOp::create(b, loc, av, offset));
        live = arith::AndIOp::create(b, loc, rightLive, vector::BroadcastOp::create(b, loc, maskType, valid));
      }
      // Mask both factors: zero times a NaN in an inactive pair must not leak.
      Value x = arith::SelectOp::create(b, loc, live, lhs, noValues);
      Value y = arith::SelectOp::create(b, loc, live, rhs, noValues);
      sums.push_back(math::FmaOp::create(b, loc, x, y, loop.getRegionIterArgs()[m]));
      validity.push_back(arith::OrIOp::create(b, loc, live, loop.getRegionIterArgs()[rows + m]));
    }
    sums.append(validity);
    scf::YieldOp::create(b, loc, sums);
  }
  auto join = [&](Type element, ValueRange pieces) {
    Value flat = zero(VectorType::get({rows * cols}, element));
    for (auto [i, piece] : llvm::enumerate(pieces))
      flat = vector::InsertStridedSliceOp::create(b, loc, piece, flat,
          ArrayRef<int64_t>{int64_t(i) * cols}, ArrayRef<int64_t>{1});
    return flat;
  };
  Value result = join(b.getF32Type(), loop.getResults().take_front(rows));
  Value valid = join(b.getI1Type(), loop.getResults().drop_front(rows));
  return std::make_pair(result, valid);
}
