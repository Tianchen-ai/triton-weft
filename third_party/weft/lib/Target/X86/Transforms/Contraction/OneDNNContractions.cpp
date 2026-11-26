#include "Transforms/Contraction/OneDNNContractions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace weft::x86 {
namespace {

// Compact ABI codes translated to dnnl_data_type_t in launcher.c:
// 0 = f32, 1 = f16, 2 = bf16, 3 = s8, 4 = u8, 5 = s32.
enum BrgemmDTypeCode : int32_t {
  DT_F32 = 0,
  DT_F16 = 1,
  DT_BF16 = 2,
  DT_S8 = 3,
  DT_U8 = 4,
  DT_S32 = 5,
};

Value idx(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIndexOp::create(b, loc, n);
}
Value i32c(OpBuilder &b, Location loc, int32_t n) {
  return arith::ConstantIntOp::create(b, loc, n, 32);
}
Value i8c(OpBuilder &b, Location loc, int8_t n) {
  return arith::ConstantIntOp::create(b, loc, n, 8);
}
Value boolc(OpBuilder &b, Location loc, bool v) {
  return arith::ConstantOp::create(b, loc, b.getI1Type(), b.getBoolAttr(v));
}
Value toI64(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isInteger(64)) return v;
  return arith::IndexCastUIOp::create(b, loc, b.getI64Type(), v);
}

Type physicalScalar(MLIRContext *ctx, Type elem) {
  if (auto intTy = dyn_cast<IntegerType>(elem))
    return IntegerType::get(ctx, intTy.getWidth());
  return elem;
}

Value widenElement(OpBuilder &b, Location loc, Value v, Type srcLogical, Type dstScalar) {
  if (v.getType() == dstScalar) return v;
  if (v.getType().isBF16() && dstScalar.isInteger(16))
    return arith::BitcastOp::create(b, loc, dstScalar, v);
  if (v.getType().isBF16() && dstScalar.isF32()) {
    Value bits = arith::BitcastOp::create(b, loc, b.getI16Type(), v);
    Value wide = arith::ExtUIOp::create(b, loc, b.getI32Type(), bits);
    Value shifted = arith::ShLIOp::create(b, loc, wide, i32c(b, loc, 16));
    return arith::BitcastOp::create(b, loc, b.getF32Type(), shifted);
  }
  if (isa<FloatType>(v.getType()) && isa<FloatType>(dstScalar))
    return arith::ExtFOp::create(b, loc, dstScalar, v);
  if (auto intTy = dyn_cast<IntegerType>(srcLogical)) {
    if (intTy.isUnsigned())
      return arith::ExtUIOp::create(b, loc, dstScalar, v);
    return arith::ExtSIOp::create(b, loc, dstScalar, v);
  }
  return v;
}

Value widenVector(OpBuilder &b, Location loc, Value v, Type srcLogical, VectorType dstVec) {
  if (v.getType() == dstVec) return v;
  auto srcVec = cast<VectorType>(v.getType());
  if (srcVec.getElementType().isBF16() && dstVec.getElementType().isF32()) {
    auto i16Vec = srcVec.clone(b.getI16Type());
    auto i32Vec = dstVec.clone(b.getI32Type());
    Value bits = arith::BitcastOp::create(b, loc, i16Vec, v);
    Value wide = arith::ExtUIOp::create(b, loc, i32Vec, bits);
    Value c16 = arith::ConstantOp::create(
        b, loc, i32Vec, DenseElementsAttr::get(i32Vec, b.getI32IntegerAttr(16)));
    Value shifted = arith::ShLIOp::create(b, loc, wide, c16);
    return arith::BitcastOp::create(b, loc, dstVec, shifted);
  }
  if (isa<FloatType>(srcVec.getElementType()) && isa<FloatType>(dstVec.getElementType()))
    return arith::ExtFOp::create(b, loc, dstVec, v);
  if (auto intTy = dyn_cast<IntegerType>(srcLogical)) {
    if (intTy.isUnsigned())
      return arith::ExtUIOp::create(b, loc, dstVec, v);
    return arith::ExtSIOp::create(b, loc, dstVec, v);
  }
  return v;
}

func::FuncOp ensureBrgemmExtern(ModuleOp module, OpBuilder &b, Location loc) {
  constexpr StringLiteral kExtern = "__weft_onednn_brgemm_execute";
  if (auto existing = module.lookupSymbol<func::FuncOp>(kExtern))
    return existing;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  Type i64 = b.getI64Type();
  Type i32 = b.getI32Type();
  auto fnType = b.getFunctionType(
      {ptr, ptr, ptr, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32}, {});
  auto fn = func::FuncOp::create(b, loc, kExtern, fnType);
  fn.setPrivate();
  return fn;
}

// Private borrowed-MemRef leaf helper so LocalStorage and BufferDeallocation
// see closed non-escaping lifetimes on every stack-promotable buffer.
void emitBrgemmHelperCall(
    ModuleOp module, OpBuilder &b, Location loc, const X86Target &target,
    bool ptrA, Value baseA, Value offA, unsigned elemBytesA,
    bool ptrB, Value baseB, Value offB, unsigned elemBytesB,
    Value memC, Value offC, unsigned elemBytesC,
    Value M, Value N, Value K, Value lda, Value ldb, Value ldc,
    int32_t dtA, int32_t dtB, int32_t dtC, int32_t addC) {
  ensureBrgemmExtern(module, b, loc);
  Type idxTy = b.getIndexType();
  Type i64Ty = b.getI64Type();
  Type i32Ty = b.getI32Type();
  Type llvmPtr = LLVM::LLVMPointerType::get(b.getContext());

  auto typeTag = [](Type t) -> std::string {
    if (isa<LLVM::LLVMPointerType>(t)) return "ptr";
    auto mr = cast<MemRefType>(t);
    std::string s;
    llvm::raw_string_ostream os(s);
    mr.getElementType().print(os);
    return "mr_" + os.str();
  };
  std::string name = "__weft_onednn_brgemm_leaf_" + typeTag(baseA.getType()) + "_" +
                     std::to_string(elemBytesA) + "_" + typeTag(baseB.getType()) + "_" +
                     std::to_string(elemBytesB) + "_" + typeTag(memC.getType()) + "_" +
                     std::to_string(elemBytesC);

  auto helper = module.lookupSymbol<func::FuncOp>(name);
  if (!helper) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    SmallVector<Type> argTypes{
        baseA.getType(), idxTy, baseB.getType(), idxTy, memC.getType(), idxTy,
        i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty, i32Ty};
    helper = func::FuncOp::create(b, loc, name, b.getFunctionType(argTypes, {}));
    helper.setPrivate();
    helper->setAttr("weft.x86.cpu", b.getStringAttr(target.cpu));
    helper->setAttr("weft.x86.vector_bits", b.getI32IntegerAttr(target.vectorBits));
    if (!ptrA) helper.setArgAttr(0, "weft.borrowed", b.getUnitAttr());
    if (!ptrB) helper.setArgAttr(2, "weft.borrowed", b.getUnitAttr());
    helper.setArgAttr(4, "weft.borrowed", b.getUnitAttr());
    helper->setAttr("weft.x86.brgemm_leaf", b.getUnitAttr());
    helper->setAttr("weft.x86.helper", b.getDictionaryAttr({
        b.getNamedAttr("name", b.getStringAttr(name)),
        b.getNamedAttr("noinline", b.getBoolAttr(false))}));

    Block *entry = helper.addEntryBlock();
    b.setInsertionPointToStart(entry);
    auto extractPtr = [&](Value base, Value elemOff, bool isRawPtr, unsigned bytes) -> Value {
      if (isRawPtr) {
        Value byteOff = arith::MulIOp::create(
            b, loc, arith::IndexCastUIOp::create(b, loc, i64Ty, elemOff),
            arith::ConstantIntOp::create(b, loc, bytes, 64));
        return LLVM::GEPOp::create(b, loc, llvmPtr, b.getI8Type(), base, ValueRange{byteOff});
      }
      Value addr = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, base);
      Value disp = arith::MulIOp::create(b, loc, elemOff, idx(b, loc, bytes));
      addr = arith::AddIOp::create(b, loc, addr, disp);
      Value addr64 = arith::IndexCastUIOp::create(b, loc, i64Ty, addr);
      return LLVM::IntToPtrOp::create(b, loc, llvmPtr, addr64);
    };
    Value pA = extractPtr(entry->getArgument(0), entry->getArgument(1), ptrA, elemBytesA);
    Value pB = extractPtr(entry->getArgument(2), entry->getArgument(3), ptrB, elemBytesB);
    Value pC = extractPtr(entry->getArgument(4), entry->getArgument(5), false, elemBytesC);
    auto externFn = ensureBrgemmExtern(module, b, loc);
    SmallVector<Value> callArgs{pA, pB, pC};
    for (unsigned i = 6; i < 16; ++i) callArgs.push_back(entry->getArgument(i));
    func::CallOp::create(b, loc, externFn, callArgs);
    func::ReturnOp::create(b, loc);
  }

  func::CallOp::create(
      b, loc, helper,
      ValueRange{baseA, offA, baseB, offB, memC, offC,
                 toI64(b, loc, M), toI64(b, loc, N), toI64(b, loc, K),
                 toI64(b, loc, lda), toI64(b, loc, ldb), toI64(b, loc, ldc),
                 i32c(b, loc, dtA), i32c(b, loc, dtB), i32c(b, loc, dtC), i32c(b, loc, addC)});
}

// A row along K already contains complete VNNI words. Transpose a square
// of such words in registers before writing consecutive columns for BRGEMM.
void packTransposedWords(OpBuilder &b, Location loc,
                         const DirectContractionView &src, Value nStride,
                         Value dest, Type element, int64_t pack, int64_t lanes,
                         Value kBlocks, Value nCols) {
  OpBuilder::InsertionGuard guard(b);
  Value zero = idx(b, loc, 0), step = idx(b, loc, lanes),
        group = idx(b, loc, pack);
  auto kg = scf::ForOp::create(b, loc, zero, kBlocks, step);
  b.setInsertionPointToStart(kg.getBody());
  Value k = arith::MulIOp::create(b, loc, kg.getInductionVar(), group);
  auto ng = scf::ForOp::create(b, loc, zero, nCols, step);
  b.setInsertionPointToStart(ng.getBody());
  auto words = VectorType::get({lanes}, b.getI32Type());
  auto elements = VectorType::get({lanes * pack}, element);
  SmallVector<Value> rows;
  for (int64_t row = 0; row < lanes; ++row) {
    Value n =
        arith::AddIOp::create(b, loc, ng.getInductionVar(), idx(b, loc, row));
    Value offset = arith::AddIOp::create(b, loc, src.offset, k);
    offset = arith::AddIOp::create(b, loc, offset,
                                   arith::MulIOp::create(b, loc, n, nStride));
    Value p = LLVM::GEPOp::create(b, loc, src.base.getType(), element, src.base,
                                  ValueRange{toI64(b, loc, offset)});
    rows.push_back(LLVM::LoadOp::create(b, loc, words, p, 1));
  }
  for (int64_t width = 1; width < lanes; width *= 2) {
    SmallVector<int64_t> low, high;
    for (int64_t lane = 0; lane < lanes; ++lane) {
      int64_t block = lane / (2 * width) * (2 * width);
      int64_t input = (lane & width) ? lanes : 0;
      low.push_back(input + block + lane % width);
      high.push_back(low.back() + width);
    }
    for (int64_t i = 0; i < lanes; i += 2 * width)
      for (int64_t j = i; j < i + width; ++j) {
        Value a = rows[j], c = rows[j + width];
        rows[j] = vector::ShuffleOp::create(b, loc, a, c, low);
        rows[j + width] = vector::ShuffleOp::create(b, loc, a, c, high);
      }
  }
  for (int64_t j = 0; j < lanes; ++j) {
    Value row =
        arith::AddIOp::create(b, loc, kg.getInductionVar(), idx(b, loc, j));
    Value start =
        arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, row, nCols),
                              ng.getInductionVar());
    Value offset = arith::MulIOp::create(b, loc, start, group);
    Value payload = vector::BitCastOp::create(b, loc, elements, rows[j]);
    vector::StoreOp::create(b, loc, payload, dest, ValueRange{offset});
  }
}

// Interleave rows of an ordinary [K,N] view into packed words along N.
void packContiguousRows(OpBuilder &b, Location loc,
                        const DirectContractionView &src, Value kStride,
                        Value dest, Type element, int64_t pack, int64_t lanes,
                        Value kBlocks, Value nCols) {
  OpBuilder::InsertionGuard guard(b);
  Value zero = idx(b, loc, 0), group = idx(b, loc, pack);
  auto kg = scf::ForOp::create(b, loc, zero, kBlocks, idx(b, loc, 1));
  b.setInsertionPointToStart(kg.getBody());
  Value k = arith::MulIOp::create(b, loc, kg.getInductionVar(), group);
  auto ng = scf::ForOp::create(b, loc, zero, nCols, idx(b, loc, lanes));
  b.setInsertionPointToStart(ng.getBody());
  auto packet = VectorType::get({lanes}, element);
  Value n = ng.getInductionVar();
  SmallVector<Value> parts;
  for (int64_t j = 0; j < pack; ++j) {
    Value row = arith::AddIOp::create(b, loc, k, idx(b, loc, j));
    Value off = arith::AddIOp::create(b, loc, src.offset, n);
    off = arith::AddIOp::create(b, loc, off,
                                arith::MulIOp::create(b, loc, row, kStride));
    Value ptr = LLVM::GEPOp::create(b, loc, src.base.getType(), element,
                                    src.base, ValueRange{toI64(b, loc, off)});
    parts.push_back(LLVM::LoadOp::create(b, loc, packet, ptr, 1));
  }
  for (int64_t width = 1; width < pack; width *= 2) {
    SmallVector<int64_t> interleave;
    for (int64_t col = 0; col < lanes; ++col)
      for (int64_t side : {0, 1})
        for (int64_t j = 0; j < width; ++j)
          interleave.push_back((col + side * lanes) * width + j);
    for (int64_t i = 0; i < pack; i += 2 * width)
      parts[i] = vector::ShuffleOp::create(b, loc, parts[i], parts[i + width],
                                           interleave);
  }
  Value row = arith::MulIOp::create(b, loc, kg.getInductionVar(), nCols);
  Value start = arith::MulIOp::create(
      b, loc, arith::AddIOp::create(b, loc, row, n), group);
  vector::StoreOp::create(b, loc, parts.front(), dest, ValueRange{start});
}

Value loadOperandElement(OpBuilder &b, Location loc, const BrgemmOperand &op,
                         ValueRange coords) {
  if (op.view) {
    const auto &v = *op.view;
    Value off = v.offset;
    for (size_t d = 0; d < coords.size(); ++d)
      off = arith::AddIOp::create(b, loc, off, arith::MulIOp::create(b, loc, coords[d], v.strides[d]));
    Type elemTy = physicalScalar(b.getContext(), op.logicalElement);
    Type memTy = isa<BFloat16Type, Float8E4M3FNType, Float8E5M2Type>(elemTy)
                     ? Type(IntegerType::get(b.getContext(), elemTy.getIntOrFloatBitWidth()))
                     : elemTy;
    unsigned bytes = memTy.getIntOrFloatBitWidth() / 8;
    Value byteOff = arith::MulIOp::create(
        b, loc, toI64(b, loc, off), arith::ConstantIntOp::create(b, loc, bytes, 64));
    Value ptr = LLVM::GEPOp::create(b, loc, LLVM::LLVMPointerType::get(b.getContext()),
                                    b.getI8Type(), v.base, ValueRange{byteOff});
    Value loaded = LLVM::LoadOp::create(b, loc, memTy, ptr);
    return memTy == elemTy ? loaded : Value(arith::BitcastOp::create(b, loc, elemTy, loaded));
  }
  Value linear = arrayLinear(b, loc, op.array.extents, coords);
  return memref::LoadOp::create(b, loc, op.array.data, linear);
}

Value loadOperandValid(OpBuilder &b, Location loc, const BrgemmOperand &op,
                       ValueRange coords) {
  if (op.view) {
    const auto &v = *op.view;
    Value active = v.guard;
    for (size_t d = 0; d < coords.size(); ++d) {
      Value inBounds = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, coords[d], v.limits[d]);
      active = arith::AndIOp::create(b, loc, active, inBounds);
    }
    return active;
  }
  Value linear = arrayLinear(b, loc, op.array.extents, coords);
  Value byte = memref::LoadOp::create(b, loc, op.array.valid, linear);
  return arith::TruncIOp::create(b, loc, b.getI1Type(), byte);
}

Value operandIsComplete(OpBuilder &b, Location loc, const BrgemmOperand &op) {
  if (op.view) {
    const auto &v = *op.view;
    Value ok = v.guard;
    for (size_t d = 0; d < v.extents.size(); ++d) {
      Value eq = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, v.extents[d], v.limits[d]);
      ok = arith::AndIOp::create(b, loc, ok, eq);
    }
    return ok;
  }
  Value total = arraySize(b, loc, op.array.extents);
  auto loop = scf::ForOp::create(b, loc, idx(b, loc, 0), total, idx(b, loc, 1),
                                 ValueRange{boolc(b, loc, true)});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value byte = memref::LoadOp::create(b, loc, op.array.valid, loop.getInductionVar());
    Value bit = arith::TruncIOp::create(b, loc, b.getI1Type(), byte);
    scf::YieldOp::create(
        b, loc, arith::AndIOp::create(b, loc, loop.getRegionIterArgs()[0], bit).getResult());
  }
  return loop.getResult(0);
}

bool isAssumedPositive(OpBuilder &b, Value v) {
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c))) return c.isStrictlyPositive();
  Block *blk = b.getInsertionBlock();
  if (!blk) return false;
  auto stripCast = [](Value x) -> Value {
    if (auto op = x.getDefiningOp<arith::IndexCastOp>()) return op.getIn();
    if (auto op = x.getDefiningOp<arith::IndexCastUIOp>()) return op.getIn();
    return x;
  };
  Value baseV = stripCast(v);
  auto checkBlock = [&](Block *block, Block::iterator endIt) -> bool {
    for (auto it = block->begin(); it != endIt; ++it) {
      auto assume = dyn_cast<LLVM::AssumeOp>(&*it);
      if (!assume) continue;
      auto cmp = assume.getCond().getDefiningOp<arith::CmpIOp>();
      if (!cmp) continue;
      auto pred = cmp.getPredicate();
      APInt rhsC, lhsC;
      if (stripCast(cmp.getLhs()) == baseV && matchPattern(cmp.getRhs(), m_ConstantInt(&rhsC))) {
        if ((pred == arith::CmpIPredicate::sgt || pred == arith::CmpIPredicate::ugt) &&
            !rhsC.isNegative())
          return true;
        if ((pred == arith::CmpIPredicate::sge || pred == arith::CmpIPredicate::uge) &&
            rhsC.isStrictlyPositive())
          return true;
      }
      if (stripCast(cmp.getRhs()) == baseV && matchPattern(cmp.getLhs(), m_ConstantInt(&lhsC))) {
        if ((pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::ult) &&
            !lhsC.isNegative())
          return true;
        if ((pred == arith::CmpIPredicate::sle || pred == arith::CmpIPredicate::ule) &&
            lhsC.isStrictlyPositive())
          return true;
      }
    }
    return false;
  };
  if (checkBlock(blk, b.getInsertionPoint())) return true;
  if (auto fn = blk->getParentOp()->getParentOfType<func::FuncOp>()) {
    Block *entry = &fn.getBody().front();
    if (entry != blk && checkBlock(entry, entry->end())) return true;
  } else if (auto fn = dyn_cast<func::FuncOp>(blk->getParentOp())) {
    Block *entry = &fn.getBody().front();
    if (entry != blk && checkBlock(entry, entry->end())) return true;
  }
  return false;
}

} // namespace

LogicalResult materializeOneDNNBrgemm(
    ModuleOp module, OpBuilder &b, Location loc,
    const X86Target &target, ContractionEngine engine,
    const BrgemmGeometry &geom, const BrgemmOperand &lhs,
    const BrgemmOperand &rhs, const ArrayStorage &output) {
  auto lhsExtents = lhs.view ? ArrayRef<Value>(lhs.view->extents) : ArrayRef<Value>(lhs.array.extents);
  auto rhsExtents = rhs.view ? ArrayRef<Value>(rhs.view->extents) : ArrayRef<Value>(rhs.array.extents);

  Value M = geom.packed3D ? lhsExtents[0] : lhsExtents[geom.leftReductionFirst ? 1 : 0];
  Value N = geom.vectorRight ? idx(b, loc, 1)
                             : (geom.packed3D ? rhsExtents[1] : rhsExtents[geom.rightReductionFirst ? 1 : 0]);
  Value K = geom.packed3D ? Value(arith::MulIOp::create(b, loc, lhsExtents[1], idx(b, loc, geom.kBlock)))
                          : lhsExtents[geom.leftReductionFirst ? 0 : 1];

  Type targetAType, targetBType, targetCType;
  Type zeroCopyAType, zeroCopyBType;
  int32_t dtA = DT_F32, dtB = DT_F32, dtC = DT_F32;
  int64_t packKBlock = 1;
  switch (engine) {
  case ContractionEngine::OneDNNBrgemmF32:
    targetAType = targetBType = targetCType = zeroCopyAType = zeroCopyBType = b.getF32Type();
    dtA = dtB = dtC = DT_F32;
    break;
  case ContractionEngine::OneDNNBrgemmF16:
    targetAType = targetBType = zeroCopyAType = zeroCopyBType = b.getF16Type();
    targetCType = b.getF32Type();
    dtA = dtB = DT_F16;
    dtC = DT_F32;
    break;
  case ContractionEngine::OneDNNBrgemmBF16:
    // Store BF16 payloads as raw i16 words so x86 payload legalization never sees unlowered bf16 memrefs.
    targetAType = targetBType = b.getI16Type();
    zeroCopyAType = zeroCopyBType = b.getBF16Type();
    targetCType = b.getF32Type();
    dtA = dtB = DT_BF16;
    dtC = DT_F32;
    packKBlock = 2;
    break;
  case ContractionEngine::OneDNNBrgemmS8:
    targetAType = targetBType = zeroCopyAType = zeroCopyBType = b.getI8Type();
    targetCType = b.getI32Type();
    dtA = dtB = DT_S8;
    dtC = DT_S32;
    packKBlock = 4;
    break;
  case ContractionEngine::OneDNNBrgemmU8S8:
    targetAType = targetBType = zeroCopyAType = zeroCopyBType = b.getI8Type();
    targetCType = b.getI32Type();
    dtA = DT_U8;
    dtB = DT_S8;
    dtC = DT_S32;
    packKBlock = 4;
    break;
  default:
    return failure();
  }
  unsigned bytesA = targetAType.getIntOrFloatBitWidth() / 8;
  unsigned bytesB = targetBType.getIntOrFloatBitWidth() / 8;
  unsigned bytesC = targetCType.getIntOrFloatBitWidth() / 8;

  Value zeroIdx = idx(b, loc, 0);
  Value oneIdx = idx(b, loc, 1);
  auto posCheck = [&](Value dim) -> Value {
    if (isAssumedPositive(b, dim)) return boolc(b, loc, true);
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, dim, zeroIdx);
  };
  Value nonEmpty = arith::AndIOp::create(
      b, loc,
      arith::AndIOp::create(b, loc, posCheck(M), posCheck(N)),
      posCheck(K));
  Value fastPath = arith::AndIOp::create(
      b, loc, nonEmpty,
      arith::AndIOp::create(b, loc, operandIsComplete(b, loc, lhs),
                            operandIsComplete(b, loc, rhs)));

  auto branch = scf::IfOp::create(b, loc, TypeRange{}, fastPath, true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&branch.getThenRegion().front());

    // Helper to pack/copy A into contiguous [M, K] when A cannot be forwarded zero-copy.
    auto materializeLocalA = [&]() -> std::pair<Value, Value> {
      Value countA = arith::MulIOp::create(b, loc, M, K);
      auto allocA = memref::AllocOp::create(
          b, loc, MemRefType::get({ShapedType::kDynamic}, targetAType), countA);
      allocA->setAttr("alignment", b.getI64IntegerAttr(64));
      auto mLoop = scf::ForOp::create(b, loc, zeroIdx, M, oneIdx);
      {
        OpBuilder::InsertionGuard mg(b);
        b.setInsertionPointToStart(mLoop.getBody());
        Value m = mLoop.getInductionVar();
        auto kLoop = scf::ForOp::create(b, loc, zeroIdx, K, oneIdx);
        b.setInsertionPointToStart(kLoop.getBody());
        Value k = kLoop.getInductionVar();
        SmallVector<Value> coords;
        if (geom.packed3D) {
          Value kb = idx(b, loc, geom.kBlock);
          coords = {m, arith::DivUIOp::create(b, loc, k, kb),
                    arith::RemUIOp::create(b, loc, k, kb)};
        } else if (geom.leftReductionFirst) {
          coords = {k, m};
        } else {
          coords = {m, k};
        }
        Value val = loadOperandElement(b, loc, lhs, coords);
        val = widenElement(b, loc, val, lhs.logicalElement, targetAType);
        Value dst = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, K), k);
        memref::StoreOp::create(b, loc, val, allocA, dst);
      }
      return {allocA, K};
    };

    // Helper to pack/copy B into [K, N] (no_trans) or [K/packKBlock, align16(N), packKBlock] (pack32).
    auto materializeLocalB = [&]() -> std::pair<Value, Value> {
      if (packKBlock == 1) {
        Value countB = arith::MulIOp::create(b, loc, K, N);
        auto allocB = memref::AllocOp::create(
            b, loc, MemRefType::get({ShapedType::kDynamic}, targetBType), countB);
        allocB->setAttr("alignment", b.getI64IntegerAttr(64));
        auto kLoop = scf::ForOp::create(b, loc, zeroIdx, K, oneIdx);
        {
          OpBuilder::InsertionGuard kg(b);
          b.setInsertionPointToStart(kLoop.getBody());
          Value k = kLoop.getInductionVar();
          auto nLoop = scf::ForOp::create(b, loc, zeroIdx, N, oneIdx);
          b.setInsertionPointToStart(nLoop.getBody());
          Value n = nLoop.getInductionVar();
          SmallVector<Value> coords;
          if (geom.vectorRight) coords = {k};
          else if (geom.packed3D) {
            Value kb = idx(b, loc, geom.kBlock);
            coords = {arith::DivUIOp::create(b, loc, k, kb), n,
                      arith::RemUIOp::create(b, loc, k, kb)};
          } else if (geom.rightReductionFirst) coords = {k, n};
          else coords = {n, k};
          Value val = loadOperandElement(b, loc, rhs, coords);
          val = widenElement(b, loc, val, rhs.logicalElement, targetBType);
          Value dst = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, k, N), n);
          memref::StoreOp::create(b, loc, val, allocB, dst);
        }
        return {allocB, N};
      }

      // VNNI pack32 layout: [K / packKBlock, ldb, packKBlock] with ldb = alignTo(N, 16).
      Value c15 = idx(b, loc, 15);
      Value c16 = idx(b, loc, 16);
      Value kbVal = idx(b, loc, packKBlock);
      Value ldb = arith::MulIOp::create(
          b, loc, arith::DivUIOp::create(b, loc, arith::AddIOp::create(b, loc, N, c15), c16), c16);
      Value kBlocks = arith::DivUIOp::create(b, loc, K, kbVal);
      Value countB = arith::MulIOp::create(
          b, loc, arith::MulIOp::create(b, loc, kBlocks, ldb), kbVal);
      auto allocB = memref::AllocOp::create(
          b, loc, MemRefType::get({ShapedType::kDynamic}, targetBType), countB);
      allocB->setAttr("alignment", b.getI64IntegerAttr(64));

      OpBuilder::InsertionGuard packGuard(b);
      int64_t lanes = std::min<int64_t>(16, target.vectorBits / 32);
      if (rhs.view && !geom.vectorRight && !geom.packed3D && lanes >= 4 &&
          llvm::isPowerOf2_64(lanes)) {
        const auto &v = *rhs.view;
        Value sk = v.strides[geom.rightReductionFirst ? 0 : 1];
        Value sn = v.strides[geom.rightReductionFirst ? 1 : 0];
        auto eq = [&](Value lhs, Value rhs) -> Value {
          return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lhs,
                                       rhs);
        };
        Value full =
            eq(arith::RemUIOp::create(b, loc, kBlocks, idx(b, loc, lanes)),
               zeroIdx);
        full = arith::AndIOp::create(b, loc, full, eq(N, ldb));
        full = arith::AndIOp::create(b, loc, full, eq(sk, oneIdx));
        auto tiled = scf::IfOp::create(b, loc, TypeRange{}, full, true);
        b.setInsertionPointToStart(tiled.thenBlock());
        packTransposedWords(b, loc, v, sn, allocB, targetBType, packKBlock,
                            lanes, kBlocks, N);
        b.setInsertionPointToStart(tiled.elseBlock());
        Value rowMajor =
            arith::AndIOp::create(b, loc, eq(sn, oneIdx), eq(N, ldb));
        auto rows = scf::IfOp::create(b, loc, TypeRange{}, rowMajor, true);
        b.setInsertionPointToStart(rows.thenBlock());
        packContiguousRows(b, loc, v, sk, allocB, targetBType, packKBlock,
                           lanes, kBlocks, N);
        b.setInsertionPointToStart(rows.elseBlock());
      }

      // Pack one complete VNNI word per column. Consecutive stores can be
      // interleaved from K rows or copied as words when K is contiguous.
      Value zeroElem = arith::ConstantOp::create(b, loc, targetBType,
                                                 b.getZeroAttr(targetBType));
      auto koLoop = scf::ForOp::create(b, loc, zeroIdx, kBlocks, oneIdx);
      {
        OpBuilder::InsertionGuard kg(b);
        b.setInsertionPointToStart(koLoop.getBody());
        Value ko = koLoop.getInductionVar();
        Value kBase = arith::MulIOp::create(b, loc, ko, kbVal);
        Value row = arith::MulIOp::create(b, loc, ko, ldb);
        auto nLoop = scf::ForOp::create(b, loc, zeroIdx, N, oneIdx);
        {
          OpBuilder::InsertionGuard ng(b);
          b.setInsertionPointToStart(nLoop.getBody());
          Value n = nLoop.getInductionVar();
          Value dst = arith::MulIOp::create(
              b, loc, arith::AddIOp::create(b, loc, row, n), kbVal);
          for (int64_t lane = 0; lane < packKBlock; ++lane) {
            Value ki = idx(b, loc, lane);
            Value k = arith::AddIOp::create(b, loc, kBase, ki);
            SmallVector<Value> coords;
            if (geom.vectorRight)
              coords = {k};
            else if (geom.packed3D) {
              Value srcKb = idx(b, loc, geom.kBlock);
              coords = {arith::DivUIOp::create(b, loc, k, srcKb), n,
                        arith::RemUIOp::create(b, loc, k, srcKb)};
            } else if (geom.rightReductionFirst)
              coords = {k, n};
            else
              coords = {n, k};
            Value val = loadOperandElement(b, loc, rhs, coords);
            val = widenElement(b, loc, val, rhs.logicalElement, targetBType);
            Value addr = arith::AddIOp::create(b, loc, dst, ki);
            memref::StoreOp::create(b, loc, val, allocB, addr);
          }
        }
        // Only the padded columns have no source. Full-width panels need no
        // memset before packing.
        Value padBegin = arith::MulIOp::create(
            b, loc, arith::AddIOp::create(b, loc, row, N), kbVal);
        Value padEnd = arith::MulIOp::create(
            b, loc, arith::AddIOp::create(b, loc, row, ldb), kbVal);
        auto pad = scf::ForOp::create(b, loc, padBegin, padEnd, oneIdx);
        b.setInsertionPointToStart(pad.getBody());
        memref::StoreOp::create(b, loc, zeroElem, allocB,
                                pad.getInductionVar());
      }
      return {allocB, ldb};
    };

    bool canTryZeroCopyA = lhs.view.has_value() &&
                           physicalScalar(b.getContext(), lhs.logicalElement) == zeroCopyAType &&
                           !geom.leftReductionFirst;
    bool canTryZeroCopyB = false;
    if (rhs.view.has_value() &&
        physicalScalar(b.getContext(), rhs.logicalElement) == zeroCopyBType) {
      if (packKBlock == 1 && !geom.packed3D && geom.rightReductionFirst)
        canTryZeroCopyB = true;
      else if (packKBlock > 1 && geom.packed3D && geom.kBlock == packKBlock)
        canTryZeroCopyB = true;
    }

    auto dispatchWithA = [&](bool ptrA, Value baseA, Value offA, Value ldaVal) {
      if (canTryZeroCopyB) {
        const auto &rv = *rhs.view;
        Value bOk;
        Value bLdb;
        if (packKBlock == 1) {
          bOk = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::eq,
              geom.vectorRight ? rv.strides[0] : rv.strides[1], oneIdx);
          bLdb = geom.vectorRight ? oneIdx : rv.strides[0];
        } else {
          Value kbVal = idx(b, loc, packKBlock);
          Value alignStride = idx(b, loc, 16 * packKBlock);
          Value s2ok = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, rv.strides[2], oneIdx);
          Value s1ok = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, rv.strides[1], kbVal);
          Value s0mod = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::eq,
              arith::RemUIOp::create(b, loc, rv.strides[0], alignStride), zeroIdx);
          Value s0ge = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::uge, rv.strides[0],
              arith::MulIOp::create(b, loc, N, kbVal));
          bOk = arith::AndIOp::create(
              b, loc, arith::AndIOp::create(b, loc, s2ok, s1ok),
              arith::AndIOp::create(b, loc, s0mod, s0ge));
          bLdb = arith::DivUIOp::create(b, loc, rv.strides[0], kbVal);
        }
        auto bBranch = scf::IfOp::create(b, loc, TypeRange{}, bOk, true);
        {
          OpBuilder::InsertionGuard bg(b);
          b.setInsertionPointToStart(&bBranch.getThenRegion().front());
          emitBrgemmHelperCall(module, b, loc, target, ptrA, baseA, offA, bytesA,
                               true, rv.base, rv.offset, bytesB,
                               output.data, zeroIdx, bytesC,
                               M, N, K, ldaVal, bLdb, N, dtA, dtB, dtC, 0);
          b.setInsertionPointToStart(&bBranch.getElseRegion().front());
          auto [localB, localLdb] = materializeLocalB();
          emitBrgemmHelperCall(module, b, loc, target, ptrA, baseA, offA, bytesA,
                               false, localB, zeroIdx, bytesB,
                               output.data, zeroIdx, bytesC,
                               M, N, K, ldaVal, localLdb, N, dtA, dtB, dtC, 0);
        }
      } else {
        auto [localB, localLdb] = materializeLocalB();
        emitBrgemmHelperCall(module, b, loc, target, ptrA, baseA, offA, bytesA,
                             false, localB, zeroIdx, bytesB,
                             output.data, zeroIdx, bytesC,
                             M, N, K, ldaVal, localLdb, N, dtA, dtB, dtC, 0);
      }
    };

    if (canTryZeroCopyA) {
      const auto &lv = *lhs.view;
      Value aOk;
      if (geom.packed3D) {
        Value s2ok = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lv.strides[2], oneIdx);
        Value s1ok = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::eq, lv.strides[1], idx(b, loc, geom.kBlock));
        aOk = arith::AndIOp::create(b, loc, s2ok, s1ok);
      } else {
        aOk = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lv.strides[1], oneIdx);
      }
      auto aBranch = scf::IfOp::create(b, loc, TypeRange{}, aOk, true);
      {
        OpBuilder::InsertionGuard ag(b);
        b.setInsertionPointToStart(&aBranch.getThenRegion().front());
        dispatchWithA(true, lv.base, lv.offset, lv.strides[0]);
        b.setInsertionPointToStart(&aBranch.getElseRegion().front());
        auto [localA, localLda] = materializeLocalA();
        dispatchWithA(false, localA, zeroIdx, localLda);
      }
    } else if (!lhs.view && !geom.leftReductionFirst &&
               physicalScalar(b.getContext(), lhs.logicalElement) ==
                   zeroCopyAType &&
               cast<MemRefType>(lhs.array.data.getType())
                       .getElementType()
                       .getIntOrFloatBitWidth() ==
                   targetAType.getIntOrFloatBitWidth()) {
      // Complete row-major array storage is already the BRGEMM A layout.
      // BF16 values and their physical i16 payload use the same bits.
      dispatchWithA(false, lhs.array.data, zeroIdx, K);
    } else {
      auto [localA, localLda] = materializeLocalA();
      dispatchWithA(false, localA, zeroIdx, localLda);
    }

    Value totalOut = arith::MulIOp::create(b, loc, M, N);
    auto vLoop = scf::ForOp::create(b, loc, zeroIdx, totalOut, oneIdx);
    {
      OpBuilder::InsertionGuard vg(b);
      b.setInsertionPointToStart(vLoop.getBody());
      memref::StoreOp::create(b, loc, i8c(b, loc, 1), output.valid, vLoop.getInductionVar());
    }

    // Masked / empty fallback branch preserving exact Canonical pairwise validity.
    b.setInsertionPointToStart(&branch.getElseRegion().front());
    auto mLoop = scf::ForOp::create(b, loc, zeroIdx, M, oneIdx);
    {
      OpBuilder::InsertionGuard mg(b);
      b.setInsertionPointToStart(mLoop.getBody());
      Value m = mLoop.getInductionVar();
      auto nLoop = scf::ForOp::create(b, loc, zeroIdx, N, oneIdx);
      b.setInsertionPointToStart(nLoop.getBody());
      Value n = nLoop.getInductionVar();
      Value zeroAcc = arith::ConstantOp::create(b, loc, targetCType, b.getZeroAttr(targetCType));
      auto red = scf::ForOp::create(b, loc, zeroIdx, K, oneIdx,
                                    ValueRange{zeroAcc, boolc(b, loc, false)});
      {
        OpBuilder::InsertionGuard rg(b);
        b.setInsertionPointToStart(red.getBody());
        Value k = red.getInductionVar();
        SmallVector<Value> aCoords, bCoords;
        if (geom.packed3D) {
          Value kb = idx(b, loc, geom.kBlock);
          Value ko = arith::DivUIOp::create(b, loc, k, kb);
          Value ki = arith::RemUIOp::create(b, loc, k, kb);
          aCoords = {m, ko, ki};
          bCoords = {ko, n, ki};
        } else {
          aCoords = geom.leftReductionFirst ? SmallVector<Value>{k, m} : SmallVector<Value>{m, k};
          if (geom.vectorRight) bCoords = {k};
          else bCoords = geom.rightReductionFirst ? SmallVector<Value>{k, n} : SmallVector<Value>{n, k};
        }
        Value aAct = loadOperandValid(b, loc, lhs, aCoords);
        Value bAct = loadOperandValid(b, loc, rhs, bCoords);
        Value act = arith::AndIOp::create(b, loc, aAct, bAct);
        auto stepIf = scf::IfOp::create(b, loc, TypeRange{targetCType}, act, true);
        {
          OpBuilder::InsertionGuard sg(b);
          b.setInsertionPointToStart(&stepIf.getThenRegion().front());
          Value av = widenElement(b, loc, loadOperandElement(b, loc, lhs, aCoords), lhs.logicalElement, targetCType);
          Value bv = widenElement(b, loc, loadOperandElement(b, loc, rhs, bCoords), rhs.logicalElement, targetCType);
          Value next = isa<FloatType>(targetCType)
                           ? Value(math::FmaOp::create(b, loc, av, bv, red.getRegionIterArgs()[0]))
                           : Value(arith::AddIOp::create(
                                 b, loc, arith::MulIOp::create(b, loc, av, bv), red.getRegionIterArgs()[0]));
          scf::YieldOp::create(b, loc, next);
          b.setInsertionPointToStart(&stepIf.getElseRegion().front());
          scf::YieldOp::create(b, loc, red.getRegionIterArgs()[0]);
        }
        scf::YieldOp::create(
            b, loc,
            ValueRange{stepIf.getResult(0),
                       arith::OrIOp::create(b, loc, red.getRegionIterArgs()[1], act)});
      }
      Value outIdx = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, N), n);
      memref::StoreOp::create(b, loc, red.getResult(0), output.data, outIdx);
      Value vByte = arith::ExtUIOp::create(b, loc, b.getI8Type(), red.getResult(1));
      memref::StoreOp::create(b, loc, vByte, output.valid, outIdx);
    }
  }
  return success();
}

bool canMaterializeSIMDArrayContraction(
    const X86Target &target, const BrgemmGeometry &geom,
    Type lhsElem, Type rhsElem, Type resElem) {
  if (!geom.valid || geom.leftReductionFirst || target.vectorBits != 512)
    return false;
  auto *ctx = resElem.getContext();
  Type outScalar = physicalScalar(ctx, resElem);
  if (!outScalar.isF32() && !outScalar.isInteger(32))
    return false;
  Type aScalar = physicalScalar(ctx, lhsElem);
  Type bScalar = physicalScalar(ctx, rhsElem);
  if (outScalar.isF32() &&
      (!(aScalar.isF32() || aScalar.isF16() || aScalar.isBF16()) ||
       !(bScalar.isF32() || bScalar.isF16() || bScalar.isBF16())))
    return false;
  if (outScalar.isInteger(32) && (!aScalar.isInteger(8) || !bScalar.isInteger(8)))
    return false;
  return true;
}

bool materializeSIMDArrayContraction(
    OpBuilder &b, Location loc, const X86Target &target,
    const BrgemmGeometry &geom, const ArrayStorage &lhs, Type lhsElem,
    const ArrayStorage &rhs, Type rhsElem,
    const ArrayStorage &output, Type resElem) {
  if (!canMaterializeSIMDArrayContraction(target, geom, lhsElem, rhsElem, resElem))
    return false;
  Type outScalar = physicalScalar(b.getContext(), resElem);
  Type aScalar = physicalScalar(b.getContext(), lhsElem);
  Type bScalar = physicalScalar(b.getContext(), rhsElem);

  Value zeroIdx = idx(b, loc, 0);
  Value oneIdx = idx(b, loc, 1);
  Value M = lhs.extents[0];
  Value K = geom.packed3D ? Value(arith::MulIOp::create(b, loc, lhs.extents[1], idx(b, loc, geom.kBlock)))
                          : lhs.extents[1];
  Value N = geom.vectorRight ? oneIdx
                             : (geom.packed3D ? rhs.extents[1] : rhs.extents[geom.rightReductionFirst ? 1 : 0]);

  BrgemmOperand lhsOp{std::nullopt, lhs, lhsElem};
  BrgemmOperand rhsOp{std::nullopt, rhs, rhsElem};
  Value complete = arith::AndIOp::create(
      b, loc,
      arith::AndIOp::create(
          b, loc,
          arith::AndIOp::create(b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, M, zeroIdx),
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, N, zeroIdx)),
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, K, zeroIdx)),
      arith::AndIOp::create(b, loc, operandIsComplete(b, loc, lhsOp),
                            operandIsComplete(b, loc, rhsOp)));

  constexpr int64_t kLanes = 16;
  auto vecOutTy = VectorType::get({kLanes}, outScalar);
  Value zeroVec = arith::ConstantOp::create(b, loc, vecOutTy, b.getZeroAttr(vecOutTy));
  Value zeroSca = arith::ConstantOp::create(b, loc, outScalar, b.getZeroAttr(outScalar));
  Value lanesVal = idx(b, loc, kLanes);

  auto branch = scf::IfOp::create(b, loc, TypeRange{}, complete, true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&branch.getThenRegion().front());

    if (!geom.packed3D && geom.rightReductionFirst && !geom.vectorRight) {
      // [M, K] x [K, N]: 4x16 register-blocked ZMM outer product + 1x16 M-tail + scalar N-tail.
      Value nVecEnd = arith::SubIOp::create(b, loc, N, arith::RemUIOp::create(b, loc, N, lanesVal));
      Value c4 = idx(b, loc, 4);
      Value mBlkEnd = arith::SubIOp::create(b, loc, M, arith::RemUIOp::create(b, loc, M, c4));

      auto nLoop = scf::ForOp::create(b, loc, zeroIdx, nVecEnd, lanesVal);
      {
        OpBuilder::InsertionGuard ng(b);
        b.setInsertionPointToStart(nLoop.getBody());
        Value n0 = nLoop.getInductionVar();

        auto mLoop4 = scf::ForOp::create(b, loc, zeroIdx, mBlkEnd, c4);
        {
          OpBuilder::InsertionGuard mg(b);
          b.setInsertionPointToStart(mLoop4.getBody());
          Value m0 = mLoop4.getInductionVar();
          auto kLoop = scf::ForOp::create(
              b, loc, zeroIdx, K, oneIdx, ValueRange{zeroVec, zeroVec, zeroVec, zeroVec});
          {
            OpBuilder::InsertionGuard kg(b);
            b.setInsertionPointToStart(kLoop.getBody());
            Value k = kLoop.getInductionVar();
            Value bOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, k, N), n0);
            Value bRaw = vector::LoadOp::create(b, loc, VectorType::get({kLanes}, bScalar), rhs.data, bOff);
            Value bVec = widenVector(b, loc, bRaw, rhsElem, vecOutTy);
            SmallVector<Value> nextAccs;
            for (int r = 0; r < 4; ++r) {
              Value mr = r == 0 ? m0 : Value(arith::AddIOp::create(b, loc, m0, idx(b, loc, r)));
              Value aOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, mr, K), k);
              Value aSca = widenElement(b, loc, memref::LoadOp::create(b, loc, lhs.data, aOff), lhsElem, outScalar);
              Value aVec = vector::BroadcastOp::create(b, loc, vecOutTy, aSca);
              Value acc = kLoop.getRegionIterArgs()[r];
              nextAccs.push_back(outScalar.isF32()
                  ? Value(math::FmaOp::create(b, loc, aVec, bVec, acc))
                  : Value(arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, aVec, bVec), acc)));
            }
            scf::YieldOp::create(b, loc, nextAccs);
          }
          for (int r = 0; r < 4; ++r) {
            Value mr = r == 0 ? m0 : Value(arith::AddIOp::create(b, loc, m0, idx(b, loc, r)));
            Value cOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, mr, N), n0);
            vector::StoreOp::create(b, loc, kLoop.getResult(r), output.data, cOff);
          }
        }

        auto mTail = scf::ForOp::create(b, loc, mBlkEnd, M, oneIdx);
        {
          OpBuilder::InsertionGuard mg(b);
          b.setInsertionPointToStart(mTail.getBody());
          Value m = mTail.getInductionVar();
          auto kLoop = scf::ForOp::create(b, loc, zeroIdx, K, oneIdx, ValueRange{zeroVec});
          {
            OpBuilder::InsertionGuard kg(b);
            b.setInsertionPointToStart(kLoop.getBody());
            Value k = kLoop.getInductionVar();
            Value bOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, k, N), n0);
            Value bRaw = vector::LoadOp::create(b, loc, VectorType::get({kLanes}, bScalar), rhs.data, bOff);
            Value bVec = widenVector(b, loc, bRaw, rhsElem, vecOutTy);
            Value aOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, K), k);
            Value aSca = widenElement(b, loc, memref::LoadOp::create(b, loc, lhs.data, aOff), lhsElem, outScalar);
            Value aVec = vector::BroadcastOp::create(b, loc, vecOutTy, aSca);
            Value acc = kLoop.getRegionIterArgs()[0];
            Value next = outScalar.isF32()
                ? Value(math::FmaOp::create(b, loc, aVec, bVec, acc))
                : Value(arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, aVec, bVec), acc));
            scf::YieldOp::create(b, loc, next);
          }
          Value cOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, N), n0);
          vector::StoreOp::create(b, loc, kLoop.getResult(0), output.data, cOff);
        }
      }

      // N-tail columns [nVecEnd .. N): SIMD reduction along K!
      Value kVecEnd = arith::SubIOp::create(b, loc, K, arith::RemUIOp::create(b, loc, K, lanesVal));
      auto mRem = scf::ForOp::create(b, loc, zeroIdx, M, oneIdx);
      {
        OpBuilder::InsertionGuard mg(b);
        b.setInsertionPointToStart(mRem.getBody());
        Value m = mRem.getInductionVar();
        auto nRem = scf::ForOp::create(b, loc, nVecEnd, N, oneIdx);
        b.setInsertionPointToStart(nRem.getBody());
        Value n = nRem.getInductionVar();
        auto kLoop = scf::ForOp::create(b, loc, zeroIdx, K, oneIdx, ValueRange{zeroSca});
        {
          OpBuilder::InsertionGuard kg(b);
          b.setInsertionPointToStart(kLoop.getBody());
          Value k = kLoop.getInductionVar();
          Value aIdx = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, K), k);
          Value bIdx = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, k, N), n);
          Value aVal = widenElement(
              b, loc, memref::LoadOp::create(b, loc, lhs.data, ValueRange{aIdx}),
              lhsElem, outScalar);
          Value bVal = widenElement(
              b, loc, memref::LoadOp::create(b, loc, rhs.data, ValueRange{bIdx}),
              rhsElem, outScalar);
          Value acc = kLoop.getRegionIterArgs()[0];
          Value next = outScalar.isF32()
              ? Value(math::FmaOp::create(b, loc, aVal, bVal, acc))
              : Value(arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, aVal, bVal), acc));
          scf::YieldOp::create(b, loc, next);
        }
        Value cOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, N), n);
        memref::StoreOp::create(b, loc, kLoop.getResult(0), output.data, cOff);
      }
      (void)kVecEnd;
    } else if (!geom.packed3D && (!geom.rightReductionFirst || geom.vectorRight)) {
      // [M, K] x [N, K] or [M, K] x [K]: both A and B are contiguous along K!
      Value kVecEnd = arith::SubIOp::create(b, loc, K, arith::RemUIOp::create(b, loc, K, lanesVal));
      auto mLoop = scf::ForOp::create(b, loc, zeroIdx, M, oneIdx);
      {
        OpBuilder::InsertionGuard mg(b);
        b.setInsertionPointToStart(mLoop.getBody());
        Value m = mLoop.getInductionVar();
        Value aRow = arith::MulIOp::create(b, loc, m, K);
        auto nLoop = scf::ForOp::create(b, loc, zeroIdx, N, oneIdx);
        b.setInsertionPointToStart(nLoop.getBody());
        Value n = nLoop.getInductionVar();
        Value bRow = geom.vectorRight ? zeroIdx : Value(arith::MulIOp::create(b, loc, n, K));
        auto kVec = scf::ForOp::create(b, loc, zeroIdx, kVecEnd, lanesVal, ValueRange{zeroVec});
        {
          OpBuilder::InsertionGuard kg(b);
          b.setInsertionPointToStart(kVec.getBody());
          Value k = kVec.getInductionVar();
          Value aIdx = arith::AddIOp::create(b, loc, aRow, k);
          Value bIdx = arith::AddIOp::create(b, loc, bRow, k);
          Value aRaw = vector::LoadOp::create(
              b, loc, VectorType::get({kLanes}, aScalar), lhs.data, ValueRange{aIdx});
          Value bRaw = vector::LoadOp::create(
              b, loc, VectorType::get({kLanes}, bScalar), rhs.data, ValueRange{bIdx});
          Value aVec = widenVector(b, loc, aRaw, lhsElem, vecOutTy);
          Value bVec = widenVector(b, loc, bRaw, rhsElem, vecOutTy);
          Value acc = kVec.getRegionIterArgs()[0];
          Value next = outScalar.isF32()
              ? Value(math::FmaOp::create(b, loc, aVec, bVec, acc))
              : Value(arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, aVec, bVec), acc));
          scf::YieldOp::create(b, loc, next);
        }
        Value sum = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, kVec.getResult(0));
        auto kTail = scf::ForOp::create(b, loc, kVecEnd, K, oneIdx, ValueRange{sum});
        {
          OpBuilder::InsertionGuard kg(b);
          b.setInsertionPointToStart(kTail.getBody());
          Value k = kTail.getInductionVar();
          Value aIdx = arith::AddIOp::create(b, loc, aRow, k);
          Value bIdx = arith::AddIOp::create(b, loc, bRow, k);
          Value aVal = widenElement(
              b, loc, memref::LoadOp::create(b, loc, lhs.data, ValueRange{aIdx}),
              lhsElem, outScalar);
          Value bVal = widenElement(
              b, loc, memref::LoadOp::create(b, loc, rhs.data, ValueRange{bIdx}),
              rhsElem, outScalar);
          Value acc = kTail.getRegionIterArgs()[0];
          Value next = outScalar.isF32()
              ? Value(math::FmaOp::create(b, loc, aVal, bVal, acc))
              : Value(arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, aVal, bVal), acc));
          scf::YieldOp::create(b, loc, next);
        }
        Value cOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, N), n);
        memref::StoreOp::create(b, loc, kTail.getResult(0), output.data, cOff);
      }
    } else {
      // 3D packed [M, K_outer, kBlock] x [K_outer, N, kBlock]
      Value kOuter = lhs.extents[1];
      Value kbVal = idx(b, loc, geom.kBlock);
      auto mLoop = scf::ForOp::create(b, loc, zeroIdx, M, oneIdx);
      {
        OpBuilder::InsertionGuard mg(b);
        b.setInsertionPointToStart(mLoop.getBody());
        Value m = mLoop.getInductionVar();
        auto nLoop = scf::ForOp::create(b, loc, zeroIdx, N, oneIdx);
        b.setInsertionPointToStart(nLoop.getBody());
        Value n = nLoop.getInductionVar();
        auto koLoop = scf::ForOp::create(b, loc, zeroIdx, kOuter, oneIdx, ValueRange{zeroSca});
        {
          OpBuilder::InsertionGuard kg(b);
          b.setInsertionPointToStart(koLoop.getBody());
          Value ko = koLoop.getInductionVar();
          Value acc = koLoop.getRegionIterArgs()[0];
          for (int64_t ki = 0; ki < geom.kBlock; ++ki) {
            Value kiVal = idx(b, loc, ki);
            Value aOff = arith::AddIOp::create(
                b, loc,
                arith::MulIOp::create(b, loc,
                    arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, kOuter), ko), kbVal),
                kiVal);
            Value bOff = arith::AddIOp::create(
                b, loc,
                arith::MulIOp::create(b, loc,
                    arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, ko, N), n), kbVal),
                kiVal);
            Value aVal = widenElement(b, loc, memref::LoadOp::create(b, loc, lhs.data, aOff), lhsElem, outScalar);
            Value bVal = widenElement(b, loc, memref::LoadOp::create(b, loc, rhs.data, bOff), rhsElem, outScalar);
            acc = outScalar.isF32()
                ? Value(math::FmaOp::create(b, loc, aVal, bVal, acc))
                : Value(arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, aVal, bVal), acc));
          }
          scf::YieldOp::create(b, loc, acc);
        }
        Value cOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, N), n);
        memref::StoreOp::create(b, loc, koLoop.getResult(0), output.data, cOff);
      }
    }

    Value totalOut = arith::MulIOp::create(b, loc, M, N);
    auto vLoop = scf::ForOp::create(b, loc, zeroIdx, totalOut, oneIdx);
    {
      OpBuilder::InsertionGuard vg(b);
      b.setInsertionPointToStart(vLoop.getBody());
      memref::StoreOp::create(b, loc, i8c(b, loc, 1), output.valid, vLoop.getInductionVar());
    }

    // Masked / empty fallback
    b.setInsertionPointToStart(&branch.getElseRegion().front());
    auto mLoop = scf::ForOp::create(b, loc, zeroIdx, M, oneIdx);
    {
      OpBuilder::InsertionGuard mg(b);
      b.setInsertionPointToStart(mLoop.getBody());
      Value m = mLoop.getInductionVar();
      auto nLoop = scf::ForOp::create(b, loc, zeroIdx, N, oneIdx);
      b.setInsertionPointToStart(nLoop.getBody());
      Value n = nLoop.getInductionVar();
      auto red = scf::ForOp::create(b, loc, zeroIdx, K, oneIdx,
                                    ValueRange{zeroSca, boolc(b, loc, false)});
      {
        OpBuilder::InsertionGuard rg(b);
        b.setInsertionPointToStart(red.getBody());
        Value k = red.getInductionVar();
        SmallVector<Value> aCoords, bCoords;
        if (geom.packed3D) {
          Value kb = idx(b, loc, geom.kBlock);
          Value ko = arith::DivUIOp::create(b, loc, k, kb);
          Value ki = arith::RemUIOp::create(b, loc, k, kb);
          aCoords = {m, ko, ki};
          bCoords = {ko, n, ki};
        } else {
          aCoords = {m, k};
          if (geom.vectorRight) bCoords = {k};
          else bCoords = geom.rightReductionFirst ? SmallVector<Value>{k, n} : SmallVector<Value>{n, k};
        }
        Value act = arith::AndIOp::create(
            b, loc, loadOperandValid(b, loc, lhsOp, aCoords), loadOperandValid(b, loc, rhsOp, bCoords));
        auto stepIf = scf::IfOp::create(b, loc, TypeRange{outScalar}, act, true);
        {
          OpBuilder::InsertionGuard sg(b);
          b.setInsertionPointToStart(&stepIf.getThenRegion().front());
          Value av = widenElement(b, loc, loadOperandElement(b, loc, lhsOp, aCoords), lhsElem, outScalar);
          Value bv = widenElement(b, loc, loadOperandElement(b, loc, rhsOp, bCoords), rhsElem, outScalar);
          Value next = outScalar.isF32()
              ? Value(math::FmaOp::create(b, loc, av, bv, red.getRegionIterArgs()[0]))
              : Value(arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, av, bv), red.getRegionIterArgs()[0]));
          scf::YieldOp::create(b, loc, next);
          b.setInsertionPointToStart(&stepIf.getElseRegion().front());
          scf::YieldOp::create(b, loc, red.getRegionIterArgs()[0]);
        }
        scf::YieldOp::create(
            b, loc,
            ValueRange{stepIf.getResult(0),
                       arith::OrIOp::create(b, loc, red.getRegionIterArgs()[1], act)});
      }
      Value cOff = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, m, N), n);
      memref::StoreOp::create(b, loc, red.getResult(0), output.data, cOff);
      Value vByte = arith::ExtUIOp::create(b, loc, b.getI8Type(), red.getResult(1));
      memref::StoreOp::create(b, loc, vByte, output.valid, cOff);
    }
  }
  return true;
}

} // namespace weft::x86
