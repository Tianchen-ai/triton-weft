#include "Weft/Target/X86Compiler.h"
#include "Weft/Target/X86Target.h"
#include "Weft/Analysis/IntegerFacts.h"
#include "IntegerReductions.h"
#include "IndexedValues.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace {

struct NarrowInteger {
  Value value;
  unsigned width;
  bool isUnsigned;
};

std::optional<weft::analysis::IntegerRange> numericRange(Value value, IntegerType type) {
  using Kind = weft::analysis::IntegerExpressionKind;
  return weft::analysis::integerRange(value, type, [](Value value)
      -> std::optional<weft::analysis::IntegerExpression> {
    auto call = value.getDefiningOp<func::CallOp>();
    if (!call) return std::nullopt;
    auto element = weft::analysis::integerElementType(value.getType());
    if (!element) return std::nullopt;
    auto signedType = [&](unsigned bits) { return IntegerType::get(value.getContext(), bits, IntegerType::Signed); };
    auto unsignedType = [&](unsigned bits) { return IntegerType::get(value.getContext(), bits, IntegerType::Unsigned); };
    weft::analysis::IntegerExpression result{signedType(element.getWidth())};
    if (call.getCallee() == "llvm.x86.avx2.pshuf.b" || call.getCallee() == "llvm.x86.avx512.pshuf.b.512") {
      result.type = unsignedType(element.getWidth()); result.kind = Kind::ZeroOr;
      result.inputs = {call.getOperand(0)}; result.interpretations = {result.type};
    } else if (call.getCallee() == "llvm.x86.avx512.vpdpbusd.128" ||
               call.getCallee() == "llvm.x86.avx512.vpdpbusd.256" ||
               call.getCallee() == "llvm.x86.avx512.vpdpbusd.512") {
      result.kind = Kind::ProductSum; result.group = 4;
      result.inputs = {call.getOperand(1), call.getOperand(2), call.getOperand(0)};
      result.interpretations = {unsignedType(8), signedType(8), signedType(32)};
    } else if (call.getCallee() == "llvm.x86.ssse3.pmadd.ub.sw.128" ||
               call.getCallee() == "llvm.x86.avx2.pmadd.ub.sw" ||
               call.getCallee() == "llvm.x86.avx512.pmaddubs.w.512") {
      result.kind = Kind::ProductSum; result.group = 2;
      result.inputs = {call.getOperand(0), call.getOperand(1)};
      result.interpretations = {unsignedType(8), signedType(8)};
    } else if (call.getCallee() == "llvm.x86.sse2.pmadd.wd" ||
               call.getCallee() == "llvm.x86.avx2.pmadd.wd" ||
               call.getCallee() == "llvm.x86.avx512.pmaddw.d.512") {
      result.kind = Kind::ProductSum; result.group = 2;
      result.inputs = {call.getOperand(0), call.getOperand(1)};
      result.interpretations = {signedType(16), signedType(16)};
    } else return std::nullopt;
    return result;
  });
}

uint64_t unsignedMaximum(Value value, unsigned width) {
  auto type = IntegerType::get(value.getContext(), width, IntegerType::Unsigned);
  auto range = numericRange(value, type);
  return range ? range->maximum.getZExtValue() : llvm::maskTrailingOnes<uint64_t>(width);
}

bool exactSaturatingPairs(Value lhs, Value rhs, IntegerType left, IntegerType right) {
  auto a = numericRange(lhs, left), c = numericRange(rhs, right);
  auto accumulator = IntegerType::get(lhs.getContext(), 16, IntegerType::Signed);
  return a && c && weft::analysis::saturatingProductSumIsExact(*a, *c, 2, accumulator);
}

void foldProvenIntegerRestorations(ModuleOp module, IRRewriter &b) {
  SmallVector<Operation *> extensions;
  module.walk([&](Operation *op) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp>(op)) extensions.push_back(op);
  });
  for (Operation *op : extensions) {
    auto truncate = op->getOperand(0).getDefiningOp<arith::TruncIOp>();
    if (!truncate || truncate.getIn().getType() != op->getResult(0).getType()) continue;
    auto source = weft::analysis::integerElementType(truncate.getIn().getType());
    auto narrow = weft::analysis::integerElementType(truncate.getType());
    auto sign = isa<arith::ExtUIOp>(op) ? IntegerType::Unsigned : IntegerType::Signed;
    auto range = numericRange(truncate.getIn(), IntegerType::get(op->getContext(), source.getWidth(), sign));
    if (range && range->fits(IntegerType::get(op->getContext(), narrow.getWidth(), sign)))
      b.replaceOp(op, truncate.getIn());
  }
}

std::optional<NarrowInteger> extension(Value value) {
  bool unsignedInput = value.getDefiningOp<arith::ExtUIOp>() != nullptr;
  auto *op = value.getDefiningOp();
  if (!op || !isa<arith::ExtSIOp, arith::ExtUIOp>(op)) return {};
  auto type = dyn_cast<VectorType>(op->getOperand(0).getType());
  if (!type || type.isScalable() || type.getRank() != 1) return {};
  return NarrowInteger{op->getOperand(0), type.getElementTypeBitWidth(), unsignedInput};
}

Value integerIntrinsic(ModuleOp module, OpBuilder &b, Location loc,
                       StringRef name, VectorType type, ValueRange operands) {
  auto declaration = module.lookupSymbol<func::FuncOp>(name);
  if (!declaration) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    declaration = func::FuncOp::create(b, loc, name,
        b.getFunctionType(operands.getTypes(), TypeRange{type}));
    declaration.setPrivate();
    declaration->setAttr("weft.x86.preserves_tile_state", b.getUnitAttr());
  }
  return func::CallOp::create(b, loc, declaration, operands).getResult(0);
}

// A small unsigned multiplier partitions byte values at ceil(256*j/factor).
// AVX-512 byte comparisons avoid widening all bytes just to discard the low
// product bits. Powers of two retain LLVM's direct shift representation.
void selectHighByteProducts(ModuleOp module, IRRewriter &b) {
  SmallVector<arith::ShRUIOp> shifts;
  module.walk([&](arith::ShRUIOp op) { shifts.push_back(op); });
  for (auto shift : shifts) {
    auto type = dyn_cast<VectorType>(shift.getType());
    auto product = shift.getLhs().getDefiningOp<arith::MulIOp>();
    auto amount = shift.getRhs().getDefiningOp<arith::ConstantOp>();
    auto dense = amount ? dyn_cast<DenseIntElementsAttr>(amount.getValue()) : DenseIntElementsAttr();
    if (!type || type.getRank() != 1 || type.isScalable() || !type.getElementType().isInteger(16) ||
        !product || !dense || !dense.isSplat() || dense.getSplatValue<APInt>() != 8 ||
        unsignedMaximum(product.getLhs(), 16) > 255 || unsignedMaximum(product.getRhs(), 16) > 255)
      continue;
    auto function = shift->getParentOfType<func::FuncOp>();
    auto width = function ? function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits") : IntegerAttr();
    if (!width || width.getInt() != 512) continue;
    Value input;
    int64_t factor = 0;
    for (unsigned side = 0; side < 2; ++side) {
      auto c = product->getOperand(side).getDefiningOp<arith::ConstantOp>();
      auto values = c ? dyn_cast<DenseIntElementsAttr>(c.getValue()) : DenseIntElementsAttr();
      if (!values || !values.isSplat()) continue;
      factor = values.getSplatValue<APInt>().getZExtValue();
      input = product->getOperand(1 - side);
      break;
    }
    if (!input || factor < 2 || factor > 4 || llvm::isPowerOf2_64(factor)) continue;
    b.setInsertionPoint(shift);
    auto bytes = type.clone(b.getI8Type());
    Value value = arith::TruncIOp::create(b, shift.getLoc(), bytes, input);
    auto constant = [&](int64_t number) -> Value {
      return arith::ConstantOp::create(b, shift.getLoc(), bytes,
          DenseElementsAttr::get(bytes, b.getI8IntegerAttr(number)));
    };
    Value output = constant(0);
    for (int64_t interval = 1; interval < factor; ++interval) {
      const int64_t threshold = (256 * interval + factor - 1) / factor;
      Value predicate = arith::CmpIOp::create(b, shift.getLoc(), arith::CmpIPredicate::uge,
          value, constant(threshold));
      output = arith::SelectOp::create(b, shift.getLoc(), predicate, constant(interval), output);
    }
    b.replaceOpWithNewOp<arith::ExtUIOp>(shift, type, output);
  }
}

// Keep the free/output axis in SIMD lanes while closing independent integer
// reductions. Each output is the same modular sum as its original reduction;
// scale, cast and lifecycle consumers remain after that boundary.
void combineReductionGroups(ModuleOp module, IRRewriter &b) {
  SmallVector<vector::FromElementsOp> packs;
  module.walk([&](vector::FromElementsOp pack) { packs.push_back(pack); });
  for (auto pack : packs) {
    auto output = dyn_cast<VectorType>(pack.getType());
    if (!output || output.getRank() != 1 || output.isScalable() ||
        !output.getElementType().isInteger(32)) continue;
    int64_t groups = output.getNumElements();
    if (groups < 4 || groups > 64 || !llvm::isPowerOf2_64(groups)) continue;
    SmallVector<Value> inputs;
    int64_t width = 0;
    bool supported = true;
    for (Value value : pack.getOperands()) {
      auto reduction = value.getDefiningOp<vector::ReductionOp>();
      if (!reduction || !value.hasOneUse() || reduction->getBlock() != pack->getBlock() ||
          reduction.getKind() != vector::CombiningKind::ADD || reduction.getAcc()) {
        supported = false;
        break;
      }
      auto type = cast<VectorType>(reduction.getVector().getType());
      int64_t count = type.getNumElements();
      Operation *producer = reduction.getVector().getDefiningOp();
      if (type.getRank() != 1 || type.isScalable() || count < 2 || count > 32 ||
          !llvm::isPowerOf2_64(count) || (width && count != width) ||
          !producer || producer->getBlock() != pack->getBlock()) {
        supported = false;
        break;
      }
      width = count;
      inputs.push_back(reduction.getVector());
    }
    auto function = pack->getParentOfType<func::FuncOp>();
    auto vectorBits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits");
    if (!supported || !vectorBits) continue;
    // PHADDD combines within 128-bit lanes. Four outputs fit that lane
    // topology without permutations between every tree level. First reduce
    // each producer to at most one AVX2 register, beside that producer, rather
    // than keeping all wide partials live until the final result is assembled.
    constexpr int64_t batch = 4;
    int64_t packetWidth = width >= 8 ? 8 : 4;
    auto packetType = VectorType::get({packetWidth}, b.getI32Type());
    auto slice = [&](Location loc, Value value, int64_t begin, int64_t size) -> Value {
      SmallVector<int64_t> indices;
      for (int64_t i = 0; i < size; ++i) indices.push_back(begin + i);
      return vector::ShuffleOp::create(b, loc, value, value, indices);
    };
    for (Value &input : inputs) {
      b.setInsertionPointAfter(input.getDefiningOp());
      for (int64_t count = width; count > packetWidth; count /= 2) {
        Value low = slice(pack.getLoc(), input, 0, count / 2);
        Value high = slice(pack.getLoc(), input, count / 2, count / 2);
        input = arith::AddIOp::create(b, pack.getLoc(), low, high);
      }
      if (width < packetWidth) {
        Value padding = arith::ConstantOp::create(b, pack.getLoc(), packetType,
            DenseElementsAttr::get(packetType, b.getI32IntegerAttr(0)));
        input = vector::InsertStridedSliceOp::create(b, pack.getLoc(), input, padding,
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
      }
    }
    SmallVector<Value> completed;
    for (int64_t begin = 0; begin < groups; begin += batch) {
      SmallVector<Value> packets(inputs.begin() + begin, inputs.begin() + begin + batch);
      Operation *last = packets.front().getDefiningOp();
      for (Value value : packets)
        if (last->isBeforeInBlock(value.getDefiningOp())) last = value.getDefiningOp();
      b.setInsertionPointAfter(last);
      Location loc = pack.getLoc();
      StringRef name = packetWidth == 8 ? "llvm.x86.avx2.phadd.d" : "llvm.x86.ssse3.phadd.d.128";
      auto hadd = [&](Value a, Value c) {
        return integerIntrinsic(module, b, loc, name, packetType, ValueRange{a, c});
      };
      Value low = hadd(packets[0], packets[1]), high = hadd(packets[2], packets[3]);
      Value result = hadd(low, high);
      if (packetWidth == 8)
        result = arith::AddIOp::create(b, loc, slice(loc, result, 0, 4), slice(loc, result, 4, 4));
      completed.push_back(result);
    }
    b.setInsertionPoint(pack);
    Value result = completed.front();
    if (batch != groups) {
      result = arith::ConstantOp::create(b, pack.getLoc(), output,
          DenseElementsAttr::get(output, b.getI32IntegerAttr(0)));
      for (auto [i, packet] : llvm::enumerate(completed))
        result = vector::InsertStridedSliceOp::create(b, pack.getLoc(), packet, result,
            ArrayRef<int64_t>{int64_t(i) * batch}, ArrayRef<int64_t>{1});
    }
    b.replaceOp(pack, result);
  }
}

void coalesceByteDotMasks(ModuleOp module, IRRewriter &b) {
  SmallVector<func::CallOp> calls;
  module.walk([&](func::CallOp call) {
    if (call.getCallee().starts_with("llvm.x86.avx512.vpdpbusd.")) calls.push_back(call);
  });
  for (auto call : calls) {
    if (!matchPattern(call.getOperand(0), m_Zero())) continue;
    Value left = call.getOperand(1), right = call.getOperand(2);
    auto leftSlice = left.getDefiningOp<vector::ShuffleOp>();
    auto rightSlice = right.getDefiningOp<vector::ShuffleOp>();
    SmallVector<int64_t> selection;
    if (leftSlice && rightSlice && leftSlice.getMask() == rightSlice.getMask() &&
        cast<VectorType>(leftSlice.getV1().getType()) == cast<VectorType>(rightSlice.getV1().getType())) {
      int64_t count = cast<VectorType>(leftSlice.getV1().getType()).getNumElements();
      if (llvm::all_of(leftSlice.getMask(), [&](int64_t i) { return i >= 0 && i < count; })) {
        left = leftSlice.getV1(); right = rightSlice.getV1();
        llvm::append_range(selection, leftSlice.getMask());
      }
    }
    auto lhs = left.getDefiningOp<arith::SelectOp>();
    auto rhs = right.getDefiningOp<arith::SelectOp>();
    if (!lhs || !rhs || lhs.getCondition() != rhs.getCondition() ||
        !matchPattern(lhs.getFalseValue(), m_Zero()) || !matchPattern(rhs.getFalseValue(), m_Zero()) ||
        !weft::x86::hasUniformPredicateGroups(lhs.getCondition(), 4)) continue;
    b.setInsertionPoint(call);
    SmallVector<int64_t> lanes;
    auto type = cast<VectorType>(call.getResult(0).getType());
    if (!selection.empty()) {
      bool aligned = true;
      for (size_t i = 0; i < selection.size(); ++i)
        aligned &= selection[i] % 4 == int64_t(i % 4) && selection[i] / 4 == selection[i / 4 * 4] / 4;
      if (!aligned) continue;
    }
    for (int64_t lane = 0; lane < type.getNumElements(); ++lane)
      lanes.push_back(selection.empty() ? lane * 4 : selection[lane * 4]);
    Value mask = vector::ShuffleOp::create(b, call.getLoc(), lhs.getCondition(), lhs.getCondition(), lanes);
    auto unmasked = [&](Value data) -> Value {
      if (selection.empty()) return data;
      return vector::ShuffleOp::create(b, call.getLoc(), data, data, selection);
    };
    call.setOperand(1, unmasked(lhs.getTrueValue())); call.setOperand(2, unmasked(rhs.getTrueValue()));
    b.setInsertionPointAfter(call);
    Value selected = arith::SelectOp::create(b, call.getLoc(), mask, call.getResult(0), call.getOperand(0));
    call.getResult(0).replaceAllUsesExcept(selected, selected.getDefiningOp());
  }
}

// Both VNNI forms implement modular i32 accumulation. Select the word form
// from a proven signed-word pair product only when its target feature is
// present; no saturation or source accumulator boundary is changed.
void foldDotAccumulators(ModuleOp module, IRRewriter &b) {
  SmallVector<arith::AddIOp> additions;
  module.walk([&](arith::AddIOp add) { additions.push_back(add); });
  for (auto add : additions) {
    auto type = dyn_cast<VectorType>(add.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isInteger(32)) continue;
    for (unsigned side = 0; side < 2; ++side) {
      Value contribution = add->getOperand(side), accumulator = add->getOperand(1 - side);
      struct Packet { func::CallOp call; int64_t offset; Value mask; bool words; };
      SmallVector<Packet> packets;
      SmallVector<bool> covered(type.getNumElements(), false);
      auto take = [&](Value value, int64_t offset) {
        Value mask;
        if (auto select = value.getDefiningOp<arith::SelectOp>()) {
          if (!value.hasOneUse() || !matchPattern(select.getFalseValue(), m_Zero())) return false;
          mask = select.getCondition(); value = select.getTrueValue();
        }
        auto call = value.getDefiningOp<func::CallOp>();
        if (!call || !value.hasOneUse() || call->getBlock() != add->getBlock()) return false;
        bool words = call.getCallee() == "llvm.x86.sse2.pmadd.wd" ||
                     call.getCallee() == "llvm.x86.avx2.pmadd.wd" ||
                     call.getCallee() == "llvm.x86.avx512.pmaddw.d.512";
        if (words) {
          auto function = call->getParentOfType<func::FuncOp>();
          auto cpu = function->getAttrOfType<StringAttr>("weft.x86.cpu");
          weft::X86Target target;
          std::string error;
          unsigned bits = cast<VectorType>(call.getResult(0).getType()).getNumElements() * 32;
          if (!cpu || !weft::parseX86Target(cpu.str(), "sysv", 0, target, error) ||
              !target.hasByteDot() || (bits == 512 && !target.hasFeature("avx512vnni"))) return false;
        } else if (!call.getCallee().starts_with("llvm.x86.avx512.vpdpbusd.") ||
                   !matchPattern(call.getOperand(0), m_Zero())) return false;
        int64_t size = cast<VectorType>(value.getType()).getNumElements();
        if (offset < 0 || offset + size > type.getNumElements()) return false;
        for (int64_t i = offset; i < offset + size; ++i)
          if (covered[i]) return false;
        for (int64_t i = offset; i < offset + size; ++i) covered[i] = true;
        packets.push_back({call, offset, mask, words});
        return true;
      };
      Value rest = contribution;
      bool legal = true;
      while (auto insert = rest.getDefiningOp<vector::InsertStridedSliceOp>()) {
        if (!rest.hasOneUse() || insert.getOffsets().size() != 1 ||
            cast<IntegerAttr>(insert.getStrides()[0]).getInt() != 1 ||
            !take(insert.getValueToStore(), cast<IntegerAttr>(insert.getOffsets()[0]).getInt())) {
          legal = false; break;
        }
        rest = insert.getDest();
      }
      if (rest == contribution) legal &= take(rest, 0);
      if (!legal || !llvm::all_of(covered, [](bool used) { return used; })) continue;
      b.setInsertionPoint(add);
      Value result = arith::ConstantOp::create(b, add.getLoc(), type,
          DenseElementsAttr::get(type, b.getI32IntegerAttr(0)));
      for (auto [call, offset, mask, words] : packets) {
        auto packetType = cast<VectorType>(call.getResult(0).getType());
        SmallVector<int64_t> indices;
        for (int64_t i = 0; i < packetType.getNumElements(); ++i) indices.push_back(offset + i);
        Value carry = vector::ShuffleOp::create(b, add.getLoc(), accumulator, accumulator, indices);
        std::string name = words ? "llvm.x86.avx512.vpdpwssd." +
            std::to_string(packetType.getNumElements() * 32) : call.getCallee().str();
        Value packet = integerIntrinsic(module, b, add.getLoc(), name, packetType,
            ValueRange{carry, call.getOperand(words ? 0 : 1), call.getOperand(words ? 1 : 2)});
        if (mask) packet = arith::SelectOp::create(b, add.getLoc(), mask, packet, carry);
        if (packetType == type) result = packet;
        else result = vector::InsertStridedSliceOp::create(b, add.getLoc(), packet, result,
            ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{1});
      }
      b.replaceOp(add, result);
      break;
    }
  }
}

bool selectByteSum(ModuleOp module, vector::ReductionOp reduce, IRRewriter &b) {
  auto resultType = dyn_cast<IntegerType>(reduce.getType());
  if (reduce.getKind() != vector::CombiningKind::ADD || !resultType || resultType.getWidth() > 64) return false;
  Value value = reduce.getVector(), mask;
  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    if (!matchPattern(select.getFalseValue(), m_Zero())) return false;
    mask = select.getCondition(); value = select.getTrueValue();
  }
  auto input = extension(value);
  if (!input) {
    // A row reduction often sees a projection of one shared widening. Keep
    // the projection in byte carriers so sum selection does not expand the
    // complete producer to dwords merely to narrow it again for PSADBW.
    if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
      int64_t width = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
      SmallVector<int64_t> indices;
      Value source;
      bool complete = true;
      for (int64_t index : shuffle.getMask()) {
        if (index < 0) { complete = false; break; }
        Value selected = index < width ? shuffle.getV1() : shuffle.getV2();
        if (source && source != selected) { complete = false; break; }
        source = selected;
        indices.push_back(index < width ? index : index - width);
      }
      Value sourceMask;
      if (complete && source) {
        if (auto select = source.getDefiningOp<arith::SelectOp>()) {
          if (matchPattern(select.getFalseValue(), m_Zero())) {
            sourceMask = select.getCondition();
            source = select.getTrueValue();
          }
        }
      }
      auto original = complete && source ? extension(source) : std::nullopt;
      if (original && original->width <= 8 && indices.size() >= 8) {
        b.setInsertionPoint(reduce);
        Value projected = vector::ShuffleOp::create(b, reduce.getLoc(),
            original->value, original->value, indices);
        if (sourceMask) {
          Value projectedMask = vector::ShuffleOp::create(b, reduce.getLoc(), sourceMask, sourceMask, indices);
          mask = mask ? Value(arith::AndIOp::create(b, reduce.getLoc(), mask, projectedMask)) : projectedMask;
        }
        input = NarrowInteger{projected, original->width, original->isUnsigned};
      }
    }
  }
  if (!input || input->width > 8) return false;
  auto function = reduce->getParentOfType<func::FuncOp>();
  auto vectorBits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits");
  if (!vectorBits) return false;
  int64_t count = cast<VectorType>(input->value.getType()).getNumElements();
  if (count < 8) return false;
  int64_t bits = count <= 16 ? 128 : count <= 32 ? 256 : vectorBits.getInt();
  int64_t lanes = bits / 8;
  b.setInsertionPoint(reduce); Location loc = reduce.getLoc();
  auto constant = [&](VectorType type, int64_t number) -> Value {
    return arith::ConstantOp::create(b, loc, type,
        DenseElementsAttr::get(type, b.getIntegerAttr(type.getElementType(), number)));
  };
  auto bytesType = VectorType::get({count}, b.getI8Type());
  Value bytes = input->value;
  if (input->width != 8)
    bytes = input->isUnsigned ? Value(arith::ExtUIOp::create(b, loc, bytesType, bytes)) :
                                Value(arith::ExtSIOp::create(b, loc, bytesType, bytes));
  if (mask) bytes = arith::SelectOp::create(b, loc, mask, bytes, constant(bytesType, 0));
  // PSADBW against zero sums unsigned bytes exactly. Bias signed bytes after
  // masking, then remove one bias per original lane (not per padded lane).
  if (!input->isUnsigned) bytes = arith::XOrIOp::create(b, loc, bytes, constant(bytesType, 128));
  auto packetType = VectorType::get({lanes}, b.getI8Type());
  auto sumsType = VectorType::get({bits / 64}, b.getI64Type());
  Value sums = constant(sumsType, 0), zero = constant(packetType, 0);
  StringRef intrinsic = bits == 128 ? "llvm.x86.sse2.psad.bw" :
                        bits == 256 ? "llvm.x86.avx2.psad.bw" : "llvm.x86.avx512.psad.bw.512";
  for (int64_t begin = 0; begin < count; begin += lanes) {
    int64_t take = std::min(lanes, count - begin);
    SmallVector<int64_t> indices;
    for (int64_t i = 0; i < take; ++i) indices.push_back(begin + i);
    Value packet = vector::ShuffleOp::create(b, loc, bytes, bytes, indices);
    if (take < lanes) packet = vector::InsertStridedSliceOp::create(b, loc, packet, zero,
        ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
    Value part = integerIntrinsic(module, b, loc, intrinsic, sumsType, ValueRange{packet, zero});
    sums = arith::AddIOp::create(b, loc, sums, part);
  }
  // Close small results in dword packets. Modular narrowing commutes with
  // the remaining additions; correcting each packet before reduction keeps
  // independent row sums available to the shared SIMD group-combine pass.
  unsigned accumulatorBits = resultType.getWidth() <= 32 ? 32 : 64;
  auto accumulatorType = VectorType::get({bits / 64}, b.getIntegerType(accumulatorBits));
  if (accumulatorBits != 64) sums = arith::TruncIOp::create(b, loc, accumulatorType, sums);
  if (!input->isUnsigned) {
    SmallVector<APInt> biases;
    for (int64_t part = 0; part < bits / 64; ++part) {
      uint64_t terms = 0;
      for (int64_t begin = 0; begin < count; begin += lanes)
        terms += std::clamp<int64_t>(count - begin - part * 8, 0, 8);
      biases.push_back(APInt(accumulatorBits, terms * 128));
    }
    Value bias = arith::ConstantOp::create(b, loc, accumulatorType,
        DenseIntElementsAttr::get(accumulatorType, biases));
    sums = arith::SubIOp::create(b, loc, sums, bias);
  }
  Value result = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, sums);
  if (resultType.getWidth() != accumulatorBits) result = arith::TruncIOp::create(b, loc, resultType, result);
  if (reduce.getAcc()) result = arith::AddIOp::create(b, loc, reduce.getAcc(), result);
  b.replaceOp(reduce, result);
  return true;
}

class SelectIntegerProducts
    : public PassWrapper<SelectIntegerProducts, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectIntegerProducts)
  StringRef getArgument() const override { return "weft-x86-select-integer-products"; }
  StringRef getDescription() const override {
    return "Select exact high-byte products after physical packet projection";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    selectHighByteProducts(getOperation(), b);
  }
};

class SelectIntegerReductions
    : public PassWrapper<SelectIntegerReductions, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectIntegerReductions)
  StringRef getArgument() const override { return "weft-x86-select-integer-reductions"; }
  StringRef getDescription() const override {
    return "Select exact SIMD byte sums and grouped products from integer widening and reduction contracts";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    SmallVector<vector::ReductionOp> reductions;
    getOperation().walk([&](vector::ReductionOp op) { reductions.push_back(op); });
    for (auto reduce : reductions) {
      if (selectByteSum(getOperation(), reduce, b)) continue;
      if (reduce.getKind() != vector::CombiningKind::ADD || !reduce.getType().isInteger(32)) continue;
      Value value = reduce.getVector(), mask;
      if (auto select = value.getDefiningOp<arith::SelectOp>()) {
        if (!matchPattern(select.getFalseValue(), m_Zero())) continue;
        mask = select.getCondition();
        value = select.getTrueValue();
      }
      auto product = value.getDefiningOp<arith::MulIOp>();
      if (!product) continue;
      auto left = extension(product.getLhs()), right = extension(product.getRhs());
      if (!left || !right) continue;
      auto function = reduce->getParentOfType<func::FuncOp>();
      auto cpu = function->getAttrOfType<StringAttr>("weft.x86.cpu");
      weft::X86Target target;
      std::string error;
      if (cpu && weft::parseX86Target(cpu.str(), "sysv", 0, target, error) && target.hasByteDot()) {
        b.setInsertionPoint(reduce);
        auto loc = reduce.getLoc();
        auto type = cast<VectorType>(left->value.getType()).clone(b.getI1Type());
        Value valid = mask ? mask : Value(arith::ConstantOp::create(b, loc, type,
            DenseElementsAttr::get(type, b.getBoolAttr(true))));
        auto integerType = [&](const NarrowInteger &input) {
          return IntegerType::get(&getContext(), input.width,
              input.isUnsigned ? IntegerType::Unsigned : IntegerType::Signed);
        };
        auto selected = weft::selectX86IntegerQuads(getOperation(), b, loc,
            left->value, right->value, valid, valid, integerType(*left), integerType(*right), target);
        if (selected) {
          Value result = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, selected->value);
          if (reduce.getAcc()) result = arith::AddIOp::create(b, loc, reduce.getAcc(), result);
          b.replaceOp(reduce, result);
          continue;
        }
      }
      bool bytes = false;
      if (!left->isUnsigned && right->isUnsigned) std::swap(left, right);
      if (left->isUnsigned && !right->isUnsigned && left->width <= 8 && right->width <= 8) {
        bytes = exactSaturatingPairs(left->value, right->value,
            IntegerType::get(&getContext(), left->width, IntegerType::Unsigned),
            IntegerType::get(&getContext(), right->width, IntegerType::Signed));
      }
      auto fitsSignedWord = [](const NarrowInteger &input) {
        return input.width <= (input.isUnsigned ? 15u : 16u);
      };
      if (!bytes && (!fitsSignedWord(*left) || !fitsSignedWord(*right))) continue;
      auto vectorBits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits");
      if (!vectorBits) continue;
      unsigned elementBits = bytes ? 8 : 16;
      int64_t count = cast<VectorType>(value.getType()).getNumElements();
      if (count < 2) continue;
      int64_t bits = count * elementBits <= 128 ? 128 :
                     count * elementBits <= 256 ? 256 : vectorBits.getInt();
      int64_t lanes = bits / elementBits;
      b.setInsertionPoint(reduce);
      Location loc = reduce.getLoc();
      auto constant = [&](VectorType type, int64_t number) -> Value {
        return arith::ConstantOp::create(b, loc, type,
            DenseElementsAttr::get(type, b.getIntegerAttr(type.getElementType(), number)));
      };
      auto call = [&](StringRef name, VectorType type, ValueRange operands) -> Value {
        return integerIntrinsic(getOperation(), b, loc, name, type, operands);
      };
      auto converted = [&](NarrowInteger input) -> Value {
        auto type = cast<VectorType>(input.value.getType()).clone(b.getIntegerType(elementBits));
        Value result = input.value;
        if (input.width != elementBits)
          result = input.isUnsigned ? Value(arith::ExtUIOp::create(b, loc, type, result)) :
                                      Value(arith::ExtSIOp::create(b, loc, type, result));
        if (mask) result = arith::SelectOp::create(b, loc, mask, result, constant(type, 0));
        return result;
      };
      Value lhs = converted(*left), rhs = converted(*right);
      auto packetType = VectorType::get({lanes}, b.getIntegerType(elementBits));
      auto sumsType = VectorType::get({bits / 32}, b.getI32Type());
      Value sums = constant(sumsType, 0);
      for (int64_t start = 0; start < count; start += lanes) {
        auto packet = [&](Value input) -> Value {
          int64_t take = std::min(lanes, count - start);
          SmallVector<int64_t> indices;
          for (int64_t i = 0; i < take; ++i) indices.push_back(start + i);
          Value chunk = vector::ShuffleOp::create(b, loc, input, input, indices);
          if (take < lanes)
            chunk = vector::InsertStridedSliceOp::create(b, loc, chunk, constant(packetType, 0),
                ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
          return chunk;
        };
        Value a = packet(lhs), c = packet(rhs);
        StringRef wordName = bits == 128 ? "llvm.x86.sse2.pmadd.wd" :
                             bits == 256 ? "llvm.x86.avx2.pmadd.wd" : "llvm.x86.avx512.pmaddw.d.512";
        Value partial;
        if (bytes) {
          StringRef byteName = bits == 128 ? "llvm.x86.ssse3.pmadd.ub.sw.128" :
                               bits == 256 ? "llvm.x86.avx2.pmadd.ub.sw" : "llvm.x86.avx512.pmaddubs.w.512";
          auto pairsType = VectorType::get({lanes / 2}, b.getI16Type());
          Value pairs = call(byteName, pairsType, ValueRange{a, c});
          partial = call(wordName, sumsType, ValueRange{pairs, constant(pairsType, 1)});
        } else partial = call(wordName, sumsType, ValueRange{a, c});
        sums = arith::AddIOp::create(b, loc, sums, partial);
      }
      Value result = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, sums);
      if (reduce.getAcc()) result = arith::AddIOp::create(b, loc, reduce.getAcc(), result);
      b.replaceOp(reduce, result);
    }
    combineReductionGroups(getOperation(), b);
    coalesceByteDotMasks(getOperation(), b);
    foldDotAccumulators(getOperation(), b);
    foldProvenIntegerRestorations(getOperation(), b);
  }
};
} // namespace

static std::optional<weft::IntegerPairResult> selectByteDotGroups(
    ModuleOp module, OpBuilder &b, Location loc, Value lhs, Value rhs,
    Value lhsValid, Value rhsValid, IntegerType lhsElement,
    IntegerType rhsElement, const weft::X86Target &target) {
  if (!target.hasByteDot()) return {};
  auto type = dyn_cast<VectorType>(lhs.getType());
  if (!type || type.getRank() != 1 || type.isScalable() ||
      cast<VectorType>(rhs.getType()).getShape() != type.getShape() ||
      type.getNumElements() % 4)
    return {};
  auto leftRange = numericRange(lhs, lhsElement), rightRange = numericRange(rhs, rhsElement);
  auto signedByte = IntegerType::get(b.getContext(), 8, IntegerType::Signed);
  auto unsignedByte = IntegerType::get(b.getContext(), 8, IntegerType::Unsigned);
  auto machineElement = [&](IntegerType source, std::optional<weft::analysis::IntegerRange> range) {
    if (source.getWidth() <= 8) return source;
    if (range && range->fits(unsignedByte)) return unsignedByte;
    if (range && range->fits(signedByte)) return signedByte;
    return IntegerType();
  };
  lhsElement = machineElement(lhsElement, leftRange);
  rhsElement = machineElement(rhsElement, rightRange);
  if (!lhsElement || !rhsElement) return {};
  auto isUnsigned = [](IntegerType t) { return t.isUnsigned() || t.getWidth() == 1; };
  // Put an unsigned operand first when possible; a narrow unsigned rhs still
  // fits signed i8. Full U8 x U8 does not fit this instruction's contract.
  if (!isUnsigned(lhsElement) && isUnsigned(rhsElement)) {
    std::swap(lhs, rhs); std::swap(lhsValid, rhsValid); std::swap(lhsElement, rhsElement);
    std::swap(leftRange, rightRange);
  }
  if (isUnsigned(rhsElement) && rhsElement.getWidth() == 8) {
    if (rightRange && rightRange->fits(signedByte)) {
      rhsElement = signedByte;
    } else if (isUnsigned(lhsElement) && leftRange && leftRange->fits(signedByte)) {
      std::swap(lhs, rhs); std::swap(lhsValid, rhsValid); std::swap(lhsElement, rhsElement);
      std::swap(leftRange, rightRange);
      rhsElement = signedByte;
    } else return {};
  }
  int64_t count = type.getNumElements();
  unsigned maximumBits = target.hasFeature("avx512vnni") ? target.vectorBits : 256;
  unsigned bits = count <= 16 ? 128 : count <= 32 ? 256 : maximumBits;
  int64_t lanes = bits / 8;
  auto constant = [&](VectorType type, int64_t number) -> Value {
    return arith::ConstantOp::create(b, loc, type,
        DenseElementsAttr::get(type, b.getIntegerAttr(type.getElementType(), number)));
  };
  Value active = arith::AndIOp::create(b, loc, lhsValid, rhsValid);
  auto byteType = type.clone(b.getI8Type());
  auto convert = [&](Value data, IntegerType input) -> Value {
    unsigned width = cast<VectorType>(data.getType()).getElementTypeBitWidth();
    if (width > 8)
      data = arith::TruncIOp::create(b, loc, byteType, data);
    else if (width < 8)
      data = isUnsigned(input) ? Value(arith::ExtUIOp::create(b, loc, byteType, data)) :
                                Value(arith::ExtSIOp::create(b, loc, byteType, data));
    return arith::SelectOp::create(b, loc, active, data, constant(byteType, 0));
  };
  lhs = convert(lhs, lhsElement); rhs = convert(rhs, rhsElement);
  bool correction = !isUnsigned(lhsElement);
  if (correction) lhs = arith::XOrIOp::create(b, loc, lhs, constant(byteType, 128));
  auto resultType = VectorType::get({count / 4}, b.getI32Type());
  auto packetType = VectorType::get({lanes}, b.getI8Type());
  auto packetResult = VectorType::get({lanes / 4}, b.getI32Type());
  Value result = constant(resultType, 0);
  auto slice = [&](Value data, int64_t start, int64_t size) -> Value {
    SmallVector<int64_t> indices;
    for (int64_t i = 0; i < size; ++i) indices.push_back(start + i);
    return vector::ShuffleOp::create(b, loc, data, data, indices);
  };
  std::string name = "llvm.x86.avx512.vpdpbusd." + std::to_string(bits);
  for (int64_t start = 0; start < count; start += lanes) {
    int64_t take = std::min(lanes, count - start);
    auto packet = [&](Value data) -> Value {
      Value part = slice(data, start, take);
      if (take != lanes) part = vector::InsertStridedSliceOp::create(b, loc, part,
          constant(packetType, 0), ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
      return part;
    };
    Value a = packet(lhs), c = packet(rhs), zero = constant(packetResult, 0);
    Value partial = integerIntrinsic(module, b, loc, name, packetResult, ValueRange{zero, a, c});
    if (correction) {
      Value bias = integerIntrinsic(module, b, loc, name, packetResult,
          ValueRange{zero, constant(packetType, 128), c});
      partial = arith::SubIOp::create(b, loc, partial, bias);
    }
    if (take != lanes) partial = slice(partial, 0, take / 4);
    result = vector::InsertStridedSliceOp::create(b, loc, partial, result,
        ArrayRef<int64_t>{start / 4}, ArrayRef<int64_t>{1});
  }
  Value valid;
  for (int64_t member = 0; member < 4; ++member) {
    SmallVector<int64_t> indices;
    for (int64_t i = member; i < count; i += 4) indices.push_back(i);
    Value selected = vector::ShuffleOp::create(b, loc, active, active, indices);
    valid = valid ? Value(arith::OrIOp::create(b, loc, valid, selected)) : selected;
  }
  return weft::IntegerPairResult{result, valid};
}

std::optional<weft::IntegerPairResult> weft::selectX86IntegerPairs(
    ModuleOp module, OpBuilder &b, Location loc, Value lhs, Value rhs,
    Value lhsValid, Value rhsValid, IntegerType lhsElement,
    IntegerType rhsElement, unsigned resultBits, unsigned vectorBits) {
  if (resultBits != 16 && resultBits != 32) return {};
  auto left = dyn_cast<VectorType>(lhs.getType());
  auto right = dyn_cast<VectorType>(rhs.getType());
  if (!left || !right || left.getRank() != 1 || right.getRank() != 1 ||
      left.isScalable() || right.isScalable() || left.getShape() != right.getShape() ||
      left.getNumElements() % 2) return {};
  bool bytes = resultBits == 16;
  if (bytes) {
    if (!lhsElement.isUnsigned() && rhsElement.isUnsigned()) {
      std::swap(lhs, rhs);
      std::swap(lhsValid, rhsValid);
      std::swap(lhsElement, rhsElement);
    }
    if (!lhsElement.isUnsigned() || !rhsElement.isSigned() ||
        lhsElement.getWidth() > 8 || rhsElement.getWidth() > 8) return {};
    // The selected instruction saturates; prove that its entire input domain
    // equals the author's wrapping i16 pair sum before selecting it.
    if (!exactSaturatingPairs(lhs, rhs, lhsElement, rhsElement)) return {};
  } else {
    auto fits = [](IntegerType type) {
      return type.getWidth() <= (type.isUnsigned() ? 15u : 16u);
    };
    if (!fits(lhsElement) || !fits(rhsElement)) return {};
  }
  int64_t count = left.getNumElements();
  unsigned inputBits = bytes ? 8 : 16;
  int64_t bits = count * inputBits <= 128 ? 128 :
                 count * inputBits <= 256 ? 256 : vectorBits;
  int64_t lanes = bits / inputBits;
  Value valid = arith::AndIOp::create(b, loc, lhsValid, rhsValid);
  auto zeros = [&](VectorType type) -> Value {
    return arith::ConstantOp::create(b, loc, type,
        DenseElementsAttr::get(type, b.getIntegerAttr(type.getElementType(), 0)));
  };
  auto convert = [&](Value value, IntegerType type) -> Value {
    auto carrier = cast<VectorType>(value.getType()).clone(b.getIntegerType(inputBits));
    if (type.getWidth() != inputBits)
      value = (type.isUnsigned() || type.getWidth() == 1) ? Value(arith::ExtUIOp::create(b, loc, carrier, value)) :
                                 Value(arith::ExtSIOp::create(b, loc, carrier, value));
    return arith::SelectOp::create(b, loc, valid, value, zeros(carrier));
  };
  lhs = convert(lhs, lhsElement);
  rhs = convert(rhs, rhsElement);
  auto packetType = VectorType::get({lanes}, b.getIntegerType(inputBits));
  auto resultType = VectorType::get({count / 2}, b.getIntegerType(resultBits));
  auto packetResult = VectorType::get({lanes / 2}, b.getIntegerType(resultBits));
  auto slice = [&](Value value, int64_t begin, int64_t size) -> Value {
    SmallVector<int64_t> indices;
    for (int64_t i = 0; i < size; ++i) indices.push_back(begin + i);
    return vector::ShuffleOp::create(b, loc, value, value, indices);
  };
  Value output = zeros(resultType);
  for (int64_t start = 0; start < count; start += lanes) {
    int64_t take = std::min(lanes, count - start);
    auto packet = [&](Value value) -> Value {
      Value part = slice(value, start, take);
      if (take != lanes)
        part = vector::InsertStridedSliceOp::create(b, loc, part, zeros(packetType),
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
      return part;
    };
    StringRef name = bytes ? (bits == 128 ? "llvm.x86.ssse3.pmadd.ub.sw.128" :
                             bits == 256 ? "llvm.x86.avx2.pmadd.ub.sw" : "llvm.x86.avx512.pmaddubs.w.512") :
                            (bits == 128 ? "llvm.x86.sse2.pmadd.wd" :
                             bits == 256 ? "llvm.x86.avx2.pmadd.wd" : "llvm.x86.avx512.pmaddw.d.512");
    Value part = integerIntrinsic(module, b, loc, name, packetResult,
        ValueRange{packet(lhs), packet(rhs)});
    if (take != lanes) part = slice(part, 0, take / 2);
    output = vector::InsertStridedSliceOp::create(b, loc, part, output,
        ArrayRef<int64_t>{start / 2}, ArrayRef<int64_t>{1});
  }
  SmallVector<int64_t> even, odd;
  for (int64_t i = 0; i < count; i += 2) { even.push_back(i); odd.push_back(i + 1); }
  Value first = vector::ShuffleOp::create(b, loc, valid, valid, even);
  Value second = vector::ShuffleOp::create(b, loc, valid, valid, odd);
  return IntegerPairResult{output, arith::OrIOp::create(b, loc, first, second)};
}

std::optional<weft::IntegerPairResult> weft::selectX86IntegerQuads(
    ModuleOp module, OpBuilder &b, Location loc, Value lhs, Value rhs,
    Value lhsValid, Value rhsValid, IntegerType lhsElement,
    IntegerType rhsElement, const X86Target &target) {
  auto input = dyn_cast<VectorType>(lhs.getType());
  if (!input || input.getRank() != 1 || input.isScalable() || input.getNumElements() % 4)
    return {};
  if (auto dot = selectByteDotGroups(module, b, loc, lhs, rhs, lhsValid, rhsValid,
                                    lhsElement, rhsElement, target))
    return dot;
  // PMADDUBSW is only selected with a proof that saturation is unreachable.
  // Otherwise signed-word products retain the complete byte/word domain.
  auto pairs = selectX86IntegerPairs(module, b, loc, lhs, rhs, lhsValid, rhsValid,
                                     lhsElement, rhsElement, 16, target.vectorBits);
  if (pairs) {
    auto wide = cast<VectorType>(pairs->value.getType()).clone(b.getI32Type());
    pairs->value = arith::ExtSIOp::create(b, loc, wide, pairs->value);
  } else {
    pairs = selectX86IntegerPairs(module, b, loc, lhs, rhs, lhsValid, rhsValid,
                                  lhsElement, rhsElement, 32, target.vectorBits);
  }
  if (!pairs) return {};
  SmallVector<int64_t> even, odd;
  for (int64_t i = 0; i < input.getNumElements() / 2; i += 2) {
    even.push_back(i); odd.push_back(i + 1);
  }
  auto select = [&](Value value, ArrayRef<int64_t> indices) -> Value {
    return vector::ShuffleOp::create(b, loc, value, value, indices);
  };
  Value sum = arith::AddIOp::create(b, loc, select(pairs->value, even), select(pairs->value, odd));
  Value active = arith::OrIOp::create(b, loc, select(pairs->valid, even), select(pairs->valid, odd));
  return IntegerPairResult{sum, active};
}

std::unique_ptr<Pass> weft::createX86IntegerReductionPass() {
  return std::make_unique<SelectIntegerReductions>();
}

std::unique_ptr<Pass> weft::createX86IntegerProductPass() {
  return std::make_unique<SelectIntegerProducts>();
}
