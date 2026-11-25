#include "Transforms/Contraction/DenseOuterProduct.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/TypeQueries.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace wk = weft::kernel;

std::optional<std::pair<Value, Value>> weft::x86::emitDenseOuterProduct(
    OpBuilder &b, Operation &op, const X86Target &target, ValueRange inputs, ValueRange masks) {
  if (!isa<wk::OuterContractOp>(op) || !wk::logicalElement(op.getResult(0).getType()).isF32()) return {};
  auto left = op.getOperand(0).getType(), right = op.getOperand(1).getType();
  auto la = wk::logicalAxes(left), ra = wk::logicalAxes(right);
  auto over = op.getAttrOfType<DenseI64ArrayAttr>("over").asArrayRef();
  if (la.size() != 2 || ra.size() != 2 || over.size() != 1 || la[1] != over[0]) return {};
  bool transposed = ra[1] == over[0];
  if (!transposed && ra[0] != over[0]) return {};
  auto outputAxes = wk::logicalAxes(op.getResult(0).getType());
  if (outputAxes.size() != 2 || outputAxes[0] != la[0] || outputAxes[1] != ra[transposed ? 0 : 1])
    return {};
  for (Value input : inputs) {
    auto v = dyn_cast<VectorType>(input.getType());
    if (!v || !(v.getElementType().isF32() || v.getElementType().isF16() || v.getElementType().isBF16()))
      return {};
  }
  int64_t rows = wk::logicalShape(left)[0], depth = wk::logicalShape(left)[1];
  int64_t cols = wk::logicalShape(right)[transposed ? 0 : 1], lanes = target.vectorBits / 32;
  if (rows <= 0 || cols < lanes || depth < 4) return {};
  int64_t colVecs = llvm::divideCeil(cols, lanes);
  int64_t registers = target.vectorRegisters() / colVecs;
  if (registers < 6) return {};
  int64_t maskedRowGroup = std::min(rows, int64_t(llvm::bit_floor(uint64_t((registers - 4) / 2))));
  int64_t unmaskedRowGroup =
      std::min(rows, std::max<int64_t>(1, (target.vectorRegisters() - colVecs - 2) / colVecs));

  Location loc = op.getLoc();
  auto index = [&](int64_t v) { return arith::ConstantIndexOp::create(b, loc, v).getResult(); };
  auto zero = [&](Type t) { return arith::ConstantOp::create(b, loc, t, b.getZeroAttr(t)).getResult(); };
  Value start = index(0);
  auto snapshot = [&](Value v) {
    auto type = cast<VectorType>(v.getType());
    OpBuilder::InsertionGuard sg(b);
    if (Block *defBb = v.getParentBlock();
        defBb && defBb != op.getBlock() &&
        defBb->getParentOp()->isProperAncestor(&op)) {
      if (Operation *defOp = v.getDefiningOp())
        b.setInsertionPointAfter(defOp);
      else
        b.setInsertionPointToStart(defBb);
      Value z0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value buffer = memref::AllocOp::create(
          b, loc, MemRefType::get({type.getNumElements()}, type.getElementType()));
      vector::StoreOp::create(b, loc, v, buffer, z0);
      return buffer;
    }
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
  auto transposeRightSnapshot = [&](Value v) -> Value {
    if (!transposed) return snapshot(v);
    auto vTy = cast<VectorType>(v.getType());
    Type elemTy = vTy.getElementType();
    if (llvm::isPowerOf2_64(uint64_t(lanes)) && cols % lanes == 0 &&
        depth % lanes == 0 && (elemTy.isF32() || elemTy.isF16() || elemTy.isBF16())) {
      Value srcBuf = snapshot(v);
      Value dstBuf = memref::AllocOp::create(
          b, loc, MemRefType::get({depth * cols}, b.getF32Type()));
      auto rawTileTy = VectorType::get({lanes}, elemTy);
      auto f32TileTy = VectorType::get({lanes}, b.getF32Type());
      auto emitTransposeBlock = [&](Value kb) {
        for (int64_t nb = 0; nb < cols; nb += lanes) {
          SmallVector<Value, 16> tile(lanes);
          for (int64_t r = 0; r < lanes; ++r) {
            Value srcOff = arith::AddIOp::create(
                b, loc, index((nb + r) * depth), kb);
            Value rowVec = vector::LoadOp::create(b, loc, rawTileTy, srcBuf, srcOff);
            if (rawTileTy != f32TileTy)
              rowVec = arith::ExtFOp::create(b, loc, f32TileTy, rowVec);
            tile[r] = rowVec;
          }
          if (lanes == 16) {
            static constexpr int64_t kUnpackLo[16] = {
                0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29};
            static constexpr int64_t kUnpackHi[16] = {
                2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31};
            static constexpr int64_t kShuf44[16] = {
                0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
            static constexpr int64_t kShufEE[16] = {
                2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31};
            static constexpr int64_t kF32x4Lo[16] = {
                0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27};
            static constexpr int64_t kF32x4Hi[16] = {
                4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31};
            static constexpr int64_t kF64x4Lo[16] = {
                0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            static constexpr int64_t kF64x4Hi[16] = {
                8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
            SmallVector<Value, 16> s1(16), s2(16), s3(16), s4(16);
            for (int p = 0; p < 8; ++p) {
              s1[2 * p] = vector::ShuffleOp::create(b, loc, tile[2 * p], tile[2 * p + 1], kUnpackLo);
              s1[2 * p + 1] = vector::ShuffleOp::create(b, loc, tile[2 * p], tile[2 * p + 1], kUnpackHi);
            }
            for (int q = 0; q < 4; ++q) {
              s2[4 * q + 0] = vector::ShuffleOp::create(b, loc, s1[4 * q], s1[4 * q + 2], kShuf44);
              s2[4 * q + 1] = vector::ShuffleOp::create(b, loc, s1[4 * q], s1[4 * q + 2], kShufEE);
              s2[4 * q + 2] = vector::ShuffleOp::create(b, loc, s1[4 * q + 1], s1[4 * q + 3], kShuf44);
              s2[4 * q + 3] = vector::ShuffleOp::create(b, loc, s1[4 * q + 1], s1[4 * q + 3], kShufEE);
            }
            for (int c0 = 0; c0 < 4; ++c0) {
              s3[0 + c0] = vector::ShuffleOp::create(b, loc, s2[0 + c0], s2[4 + c0], kF32x4Lo);
              s3[4 + c0] = vector::ShuffleOp::create(b, loc, s2[0 + c0], s2[4 + c0], kF32x4Hi);
              s3[8 + c0] = vector::ShuffleOp::create(b, loc, s2[8 + c0], s2[12 + c0], kF32x4Lo);
              s3[12 + c0] = vector::ShuffleOp::create(b, loc, s2[8 + c0], s2[12 + c0], kF32x4Hi);
            }
            for (int c0 = 0; c0 < 4; ++c0) {
              s4[0 + c0] = vector::ShuffleOp::create(b, loc, s3[0 + c0], s3[8 + c0], kF64x4Lo);
              s4[8 + c0] = vector::ShuffleOp::create(b, loc, s3[0 + c0], s3[8 + c0], kF64x4Hi);
              s4[4 + c0] = vector::ShuffleOp::create(b, loc, s3[4 + c0], s3[12 + c0], kF64x4Lo);
              s4[12 + c0] = vector::ShuffleOp::create(b, loc, s3[4 + c0], s3[12 + c0], kF64x4Hi);
            }
            tile = std::move(s4);
          } else {
            for (int64_t bit = 1; bit < lanes; bit <<= 1) {
              SmallVector<int64_t, 16> maskLo(lanes), maskHi(lanes);
              for (int64_t c = 0; c < lanes; ++c) {
                maskLo[c] = (c & bit) ? (lanes + (c ^ bit)) : c;
                maskHi[c] = (c & bit) ? (lanes + c) : (c | bit);
              }
              SmallVector<Value, 16> next(lanes);
              for (int64_t r = 0; r < lanes; ++r) {
                if (r & bit) continue;
                int64_t rHi = r | bit;
                next[r] = vector::ShuffleOp::create(b, loc, tile[r], tile[rHi], maskLo);
                next[rHi] = vector::ShuffleOp::create(b, loc, tile[r], tile[rHi], maskHi);
              }
              tile = std::move(next);
            }
          }
          for (int64_t kSub = 0; kSub < lanes; ++kSub) {
            Value kRow = arith::AddIOp::create(b, loc, kb, index(kSub));
            Value dstOff = arith::AddIOp::create(
                b, loc, arith::MulIOp::create(b, loc, kRow, index(cols)),
                index(nb));
            vector::StoreOp::create(b, loc, tile[kSub], dstBuf, dstOff);
          }
        }
      };
      if (depth * cols <= 2048) {
        for (int64_t kb0 = 0; kb0 < depth; kb0 += lanes)
          emitTransposeBlock(index(kb0));
      } else {
        auto kbLoop = scf::ForOp::create(b, loc, start, index(depth), index(lanes));
        OpBuilder::InsertionGuard kg(b);
        b.setInsertionPointToStart(kbLoop.getBody());
        emitTransposeBlock(kbLoop.getInductionVar());
      }
      return dstBuf;
    }
    SmallVector<int64_t> order;
    for (int64_t k = 0; k < depth; ++k)
      for (int64_t n = 0; n < cols; ++n)
        order.push_back(n * depth + k);
    return snapshot(vector::ShuffleOp::create(b, loc, v, v, order));
  };
  auto transposeMaskSnapshot = [&](Value mask) -> Value {
    if (matchPattern(mask, m_One())) return {};
    if (!transposed) return maskSnapshot(mask);
    auto bytes = cast<VectorType>(mask.getType()).clone(b.getI8Type());
    Value srcBuf = snapshot(arith::ExtUIOp::create(b, loc, bytes, mask));
    Value dstBuf = memref::AllocOp::create(
        b, loc, MemRefType::get({depth * cols}, b.getI8Type()));
    auto kLoop = scf::ForOp::create(b, loc, start, index(depth), index(1));
    {
      OpBuilder::InsertionGuard kg(b);
      b.setInsertionPointToStart(kLoop.getBody());
      Value k = kLoop.getInductionVar();
      auto nLoop = scf::ForOp::create(b, loc, start, index(cols), index(1));
      OpBuilder::InsertionGuard ng(b);
      b.setInsertionPointToStart(nLoop.getBody());
      Value n = nLoop.getInductionVar();
      Value sOff = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, n, index(depth)), k);
      Value dOff = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, k, index(cols)), n);
      Value mByte = memref::LoadOp::create(b, loc, srcBuf, sOff);
      memref::StoreOp::create(b, loc, mByte, dstBuf, dOff);
    }
    return dstBuf;
  };
  Value a = snapshot(inputs[0]), weights = transposeRightSnapshot(inputs[1]);
  auto rowType = VectorType::get({cols}, b.getF32Type());
  auto maskType = rowType.clone(b.getI1Type());
  Value noValues = zero(rowType), noLanes = zero(maskType);
  Value allLanes = arith::ConstantOp::create(b, loc, maskType, DenseElementsAttr::get(maskType, true));
  auto accumulate = [&](int64_t first, int64_t count, Value av, Value bv) {
    bool masked = av || bv;
    int64_t kUnroll = (!masked && depth % 4 == 0) ? 4 : 1;
    SmallVector<Value> initial(count, noValues);
    if (masked) initial.append(count, noLanes);
    auto loop = scf::ForOp::create(b, loc, start, index(depth), index(kUnroll), initial);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k0 = loop.getInductionVar();
    SmallVector<Value> sums(loop.getRegionIterArgs().begin(),
                            loop.getRegionIterArgs().begin() + count);
    SmallVector<Value> validity;
    if (masked)
      validity.assign(loop.getRegionIterArgs().begin() + count,
                      loop.getRegionIterArgs().end());
    auto rhsType = rowType.clone(cast<MemRefType>(weights.getType()).getElementType());
    for (int64_t ku = 0; ku < kUnroll; ++ku) {
      Value k = ku == 0 ? k0 : arith::AddIOp::create(b, loc, k0, index(ku)).getResult();
      Value base = arith::MulIOp::create(b, loc, k, index(cols));
      Value rhs = vector::LoadOp::create(b, loc, rhsType, weights, base);
      if (rhsType != rowType) rhs = arith::ExtFOp::create(b, loc, rowType, rhs);
      Value rightLive = allLanes;
      if (bv) rightLive = arith::TruncIOp::create(b, loc, maskType,
          vector::LoadOp::create(b, loc, rowType.clone(b.getI8Type()), bv, base));
      for (int64_t m = 0; m < count; ++m) {
        Value offset = arith::AddIOp::create(b, loc, index((first + m) * depth), k);
        Value lhs = memref::LoadOp::create(b, loc, a, offset);
        if (!lhs.getType().isF32()) lhs = arith::ExtFOp::create(b, loc, b.getF32Type(), lhs);
        lhs = vector::BroadcastOp::create(b, loc, rowType, lhs);
        Value live = rightLive;
        if (av) {
          Value valid = arith::TruncIOp::create(b, loc, b.getI1Type(), memref::LoadOp::create(b, loc, av, offset));
          live = arith::AndIOp::create(b, loc, rightLive, vector::BroadcastOp::create(b, loc, maskType, valid));
        }
        // Mask both factors: zero times a NaN in an inactive pair must not leak.
        Value right = rhs;
        if (masked) {
          lhs = arith::SelectOp::create(b, loc, live, lhs, noValues);
          right = arith::SelectOp::create(b, loc, live, rhs, noValues);
          validity[m] = arith::OrIOp::create(b, loc, live, validity[m]);
        }
        sums[m] = math::FmaOp::create(b, loc, lhs, right, sums[m]);
      }
    }
    sums.append(validity);
    scf::YieldOp::create(b, loc, sums);
    return loop;
  };
  auto join = [&](Type element, ValueRange pieces) {
    Value flat = zero(VectorType::get({rows * cols}, element));
    for (auto [i, piece] : llvm::enumerate(pieces))
      flat = vector::InsertStridedSliceOp::create(b, loc, piece, flat,
          ArrayRef<int64_t>{int64_t(i) * cols}, ArrayRef<int64_t>{1});
    return flat;
  };
  auto multiply = [&](Value av, Value bv) {
    int64_t stepRows = (av || bv) ? maskedRowGroup : unmaskedRowGroup;
    SmallVector<Value> results, resultMasks;
    for (int64_t first = 0; first < rows; first += stepRows) {
      int64_t count = std::min(stepRows, rows - first);
      auto loop = accumulate(first, count, av, bv);
      llvm::append_range(results, loop.getResults().take_front(count));
      if (av || bv) llvm::append_range(resultMasks, loop.getResults().drop_front(count));
      else resultMasks.append(count, allLanes);
    }
    return std::make_pair(join(b.getF32Type(), results), join(b.getI1Type(), resultMasks));
  };
  Value complete;
  for (Value mask : masks) {
    if (matchPattern(mask, m_One())) continue;
    Value all;
    if (Block *defBb = mask.getParentBlock();
        defBb && defBb != op.getBlock() &&
        defBb->getParentOp()->isProperAncestor(&op)) {
      OpBuilder::InsertionGuard mg(b);
      if (Operation *defOp = mask.getDefiningOp())
        b.setInsertionPointAfter(defOp);
      else
        b.setInsertionPointToStart(defBb);
      all = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, mask);
    } else {
      all = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, mask);
    }
    complete = complete ? b.createOrFold<arith::AndIOp>(loc, complete, all) : all;
  }
  if (!complete) return multiply({}, {});
  // Check each input once. Interior tiles need just the FP32 accumulators;
  // mixed tiles retain pairwise masking, including nonfinite inactive lanes.
  // Write both branches into bounded stack storage so the scf.if merge does
  // not pass wide ZMM tuples by SSA value across basic blocks.
  auto shape = VectorType::get({rows * cols}, b.getF32Type());
  auto fullMaskShape = shape.clone(b.getI1Type());
  Value outBuf = memref::AllocOp::create(b, loc, MemRefType::get({rows * cols}, b.getF32Type()));
  Value maskBuf = memref::AllocOp::create(b, loc, MemRefType::get({rows * cols}, b.getI8Type()));
  auto choice = scf::IfOp::create(b, loc, TypeRange{fullMaskShape}, complete, true);
  OpBuilder::InsertionGuard insertion(b);
  b.setInsertionPointToStart(choice.thenBlock());
  for (int64_t first = 0; first < rows; first += unmaskedRowGroup) {
    int64_t count = std::min(unmaskedRowGroup, rows - first);
    auto loop = accumulate(first, count, {}, {});
    for (int64_t m = 0; m < count; ++m)
      vector::StoreOp::create(b, loc, loop.getResult(m), outBuf, index((first + m) * cols));
  }
  Value allTrueFull = arith::ConstantOp::create(
      b, loc, fullMaskShape, DenseElementsAttr::get(fullMaskShape, true));
  scf::YieldOp::create(b, loc, ValueRange{allTrueFull});
  b.setInsertionPointToStart(choice.elseBlock());
  Value av = maskSnapshot(masks[0]), bv = transposeMaskSnapshot(masks[1]);
  auto rLoop = scf::ForOp::create(b, loc, start, index(rows), index(1));
  {
    OpBuilder::InsertionGuard rg(b);
    b.setInsertionPointToStart(rLoop.getBody());
    Value rIv = rLoop.getInductionVar();
    Value rBaseK = arith::MulIOp::create(b, loc, rIv, index(depth));
    auto kLoop = scf::ForOp::create(b, loc, start, index(depth), index(1), ValueRange{noValues, noLanes});
    {
      OpBuilder::InsertionGuard kg(b);
      b.setInsertionPointToStart(kLoop.getBody());
      Value k = kLoop.getInductionVar();
      Value base = arith::MulIOp::create(b, loc, k, index(cols));
      auto rhsType = rowType.clone(cast<MemRefType>(weights.getType()).getElementType());
      Value rhs = vector::LoadOp::create(b, loc, rhsType, weights, base);
      if (rhsType != rowType) rhs = arith::ExtFOp::create(b, loc, rowType, rhs);
      Value rightLive = allLanes;
      if (bv) rightLive = arith::TruncIOp::create(b, loc, maskType,
          vector::LoadOp::create(b, loc, rowType.clone(b.getI8Type()), bv, base));
      Value offset = arith::AddIOp::create(b, loc, rBaseK, k);
      Value lhs = memref::LoadOp::create(b, loc, a, offset);
      if (!lhs.getType().isF32()) lhs = arith::ExtFOp::create(b, loc, b.getF32Type(), lhs);
      lhs = vector::BroadcastOp::create(b, loc, rowType, lhs);
      Value live = rightLive;
      if (av) {
        Value valid = arith::TruncIOp::create(b, loc, b.getI1Type(), memref::LoadOp::create(b, loc, av, offset));
        live = arith::AndIOp::create(b, loc, rightLive, vector::BroadcastOp::create(b, loc, maskType, valid));
      }
      lhs = arith::SelectOp::create(b, loc, live, lhs, noValues);
      Value right = arith::SelectOp::create(b, loc, live, rhs, noValues);
      Value nextValid = arith::OrIOp::create(b, loc, live, kLoop.getRegionIterArgs()[1]);
      Value nextSum = math::FmaOp::create(b, loc, lhs, right, kLoop.getRegionIterArgs()[0]);
      scf::YieldOp::create(b, loc, ValueRange{nextSum, nextValid});
    }
    Value rBaseC = arith::MulIOp::create(b, loc, rIv, index(cols));
    vector::StoreOp::create(b, loc, kLoop.getResult(0), outBuf, rBaseC);
    Value validBytes = arith::ExtUIOp::create(b, loc, rowType.clone(b.getI8Type()), kLoop.getResult(1));
    vector::StoreOp::create(b, loc, validBytes, maskBuf, rBaseC);
  }
  Value tailMaskBytes = vector::LoadOp::create(b, loc, shape.clone(b.getI8Type()), maskBuf, start);
  Value tailMask = arith::TruncIOp::create(b, loc, fullMaskShape, tailMaskBytes);
  scf::YieldOp::create(b, loc, ValueRange{tailMask});
  b.setInsertionPointAfter(choice);
  Value outVals = vector::LoadOp::create(b, loc, shape, outBuf, start);
  return std::make_pair(outVals, choice.getResult(0));
}
