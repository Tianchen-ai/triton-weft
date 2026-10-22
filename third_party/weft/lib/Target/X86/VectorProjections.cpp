#include "Weft/Target/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace {

bool integerLaneLocal(Operation *op, bool byteMultiply = false);

llvm::DenseSet<Operation *> nativeIntegerClosure(ModuleOp module) {
  llvm::DenseSet<Operation *> result;
  auto local = [](Operation *op) {
    return integerLaneLocal(op, true) || isa<vector::ShuffleOp, vector::InsertStridedSliceOp,
        vector::ExtractStridedSliceOp, vector::BitCastOp>(op);
  };
  module.walk([&](Operation *conversion) {
    if (!isa<arith::SIToFPOp, arith::UIToFPOp>(conversion)) return;
    auto type = dyn_cast<VectorType>(conversion->getResult(0).getType());
    auto source = dyn_cast<VectorType>(conversion->getOperand(0).getType());
    if (!type || !source || type.isScalable() || !type.getElementType().isF32() ||
        source.getElementTypeBitWidth() > 8) return;
    SmallVector<Value> inputs(conversion->getOperands());
    while (!inputs.empty()) {
      Operation *op = inputs.pop_back_val().getDefiningOp();
      if (!op || op->getBlock() != conversion->getBlock() || !local(op) ||
          !result.insert(op).second) continue;
      llvm::append_range(inputs, op->getOperands());
    }
  });
  module.walk([&](func::CallOp call) {
    StringRef name = call.getCallee();
    bool lookup = name == "llvm.x86.avx2.pshuf.b" || name == "llvm.x86.avx512.pshuf.b.512";
    bool product = name.starts_with("llvm.x86.avx512.vpdp") ||
        name == "llvm.x86.avx2.pmadd.ub.sw" || name == "llvm.x86.avx2.pmadd.wd" ||
        name == "llvm.x86.avx512.pmaddubs.w.512" || name == "llvm.x86.avx512.pmaddw.d.512";
    if (!lookup && !product) return;
    SmallVector<Value> inputs;
    if (lookup) inputs.push_back(call.getOperand(1));
    else llvm::append_range(inputs, call.getOperands());
    while (!inputs.empty()) {
      Operation *op = inputs.pop_back_val().getDefiningOp();
      if (!op || op->getBlock() != call->getBlock() || !local(op) || !result.insert(op).second) continue;
      llvm::append_range(inputs, op->getOperands());
    }
    SmallVector<Operation *> consumers(call->getUsers());
    while (!consumers.empty()) {
      Operation *op = consumers.pop_back_val();
      if (op->getBlock() != call->getBlock() || !local(op) || !result.insert(op).second) continue;
      llvm::append_range(consumers, op->getUsers());
    }
  });
  module.walk([&](vector::StoreOp store) {
    if (!store->hasAttr("weft.x86.amx_pack")) return;
    SmallVector<Value> inputs{store.getValueToStore()};
    while (!inputs.empty()) {
      Operation *op = inputs.pop_back_val().getDefiningOp();
      if (!op || op->getBlock() != store->getBlock() || !local(op) ||
          !result.insert(op).second) continue;
      llvm::append_range(inputs, op->getOperands());
    }
  });
  return result;
}

// Local Value indexing retains explicit logical validity during lowering.
// After constant folding, a complete fixed-index reconstruction is a single
// permutation of one SSA snapshot. Spell that permutation directly instead of
// relying on LLVM to discover it through hundreds of scalar insertions.
class CoalesceVectorProjections
    : public PassWrapper<CoalesceVectorProjections, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoalesceVectorProjections)
  StringRef getArgument() const override { return "weft-x86-coalesce-vector-projections"; }
  StringRef getDescription() const override {
    return "Coalesce complete constant-index vector reconstructions without changing value or mask semantics";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, LLVM::LLVMDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    SmallVector<LLVM::InsertElementOp> roots;
    getOperation().walk([&](LLVM::InsertElementOp op) {
      if (op.getRes().use_empty()) return;
      if (op.getRes().hasOneUse()) {
        auto next = dyn_cast<LLVM::InsertElementOp>(*op.getRes().getUsers().begin());
        if (next && next.getVector() == op.getRes()) return;
      }
      roots.push_back(op);
    });
    IRRewriter b(&getContext());
    for (auto root : roots) {
      auto type = root.getRes().getType();
      if (type.getRank() != 1 || type.isScalable()) continue;
      int64_t count = type.getNumElements(), missing = count;
      SmallVector<int64_t> indices(count, -1);
      Value current = root.getRes(), source;
      // Bound the analysis even for a source with repeated element updates.
      for (int64_t steps = 0; missing && steps < 2 * count; ++steps) {
        auto insert = current.getDefiningOp<LLVM::InsertElementOp>();
        APInt position;
        if (!insert || !matchPattern(insert.getPosition(), m_ConstantInt(&position)) ||
            position.getActiveBits() > 63 || position.getZExtValue() >= static_cast<uint64_t>(count)) break;
        int64_t lane = position.getZExtValue();
        if (indices[lane] < 0) {
          auto extract = insert.getValue().getDefiningOp<LLVM::ExtractElementOp>();
          APInt from;
          if (!extract || !matchPattern(extract.getPosition(), m_ConstantInt(&from)) ||
              from.getActiveBits() > 63) break;
          auto inputType = extract.getVector().getType();
          if (inputType.getRank() != 1 || inputType.isScalable() ||
              from.getZExtValue() >= static_cast<uint64_t>(inputType.getNumElements()) ||
              (source && source != extract.getVector())) break;
          source = extract.getVector();
          indices[lane] = from.getZExtValue();
          --missing;
        }
        current = insert.getVector();
      }
      if (missing) continue;
      b.setInsertionPoint(root);
      bool bfloat = type.getElementType().isBF16();
      if (bfloat)
        source = arith::BitcastOp::create(b, root.getLoc(),
            cast<VectorType>(source.getType()).clone(b.getI16Type()), source);
      Value result = vector::ShuffleOp::create(b, root.getLoc(), source, source, indices);
      if (bfloat) result = arith::BitcastOp::create(b, root.getLoc(), type, result);
      b.replaceOp(root, result);
    }
  }
};

class LegalizeSubbyteProjections
    : public PassWrapper<LegalizeSubbyteProjections, OperationPass<ModuleOp>> {
  bool selectedConsumersOnly;
  void materializeBytePlanes(IRRewriter &b) {
    SmallVector<arith::ShLIOp> shifts;
    getOperation().walk([&](arith::ShLIOp shift) { shifts.push_back(shift); });
    for (auto shift : shifts) {
      auto type = dyn_cast<VectorType>(shift.getType());
      auto widen = shift.getLhs().getDefiningOp<arith::ExtUIOp>();
      if (!type || type.isScalable() || type.getRank() != 1 ||
          !type.getElementType().isInteger(8) || type.getNumElements() % 4 ||
          !widen || !widen->hasOneUse()) continue;
      auto narrow = widen.getIn().getDefiningOp<arith::TruncIOp>();
      if (!narrow || !narrow->hasOneUse()) continue;
      unsigned width = cast<VectorType>(narrow.getType()).getElementTypeBitWidth();
      if (width >= 8) continue;
      Value input = narrow.getIn();
      if (auto mask = input.getDefiningOp<arith::AndIOp>()) {
        DenseIntElementsAttr literal;
        Value data;
        for (unsigned side = 0; side != 2; ++side)
          if (auto constant = mask->getOperand(side).getDefiningOp<arith::ConstantOp>()) {
            literal = dyn_cast<DenseIntElementsAttr>(constant.getValue());
            data = mask->getOperand(1 - side);
            if (literal) break;
          }
        if (!literal || llvm::any_of(literal.getValues<APInt>(), [&](APInt value) {
              return !value.trunc(width).isAllOnes();
            })) continue;
        input = data;
      }
      auto extract = input.getDefiningOp<arith::ShRUIOp>();
      if (!extract) continue;
      auto bytes = extract.getLhs().getDefiningOp<arith::ExtUIOp>();
      auto sourceType = bytes ? dyn_cast<VectorType>(bytes.getIn().getType()) : VectorType();
      auto from = extract.getRhs().getDefiningOp<arith::ConstantOp>();
      auto to = shift.getRhs().getDefiningOp<arith::ConstantOp>();
      auto fromBits = from ? dyn_cast<DenseIntElementsAttr>(from.getValue()) : DenseIntElementsAttr();
      auto toBits = to ? dyn_cast<DenseIntElementsAttr>(to.getValue()) : DenseIntElementsAttr();
      if (!sourceType || sourceType != type || !fromBits || !toBits) continue;
      if (auto shuffle = bytes.getIn().getDefiningOp<vector::ShuffleOp>()) {
        auto function = shift->getParentOfType<func::FuncOp>();
        auto bits = function ? function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits") : IntegerAttr();
        if (!bits) continue;
        int64_t packet = bits.getInt() / 8;
        int64_t firstSize = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
        bool local = true;
        for (int64_t first = 0; first < type.getNumElements(); first += packet) {
          Value source;
          int64_t sourcePacket = -1;
          for (int64_t lane = first; lane < std::min(first + packet, type.getNumElements()); ++lane) {
            int64_t index = shuffle.getMask()[lane];
            if (index < 0) { local = false; break; }
            Value operand = index < firstSize ? shuffle.getV1() : shuffle.getV2();
            int64_t position = index < firstSize ? index : index - firstSize;
            if (source && (source != operand || sourcePacket != position / packet)) { local = false; break; }
            source = operand;
            sourcePacket = position / packet;
          }
          if (!local) break;
        }
        // A word plane is wider than its predicate carrier. Keep the original
        // representation when constructing it would mix several source packets
        // for each output packet and extend their residency across consumers.
        if (!local) continue;
      }
      SmallVector<APInt> starts(fromBits.getValues<APInt>()), ends(toBits.getValues<APInt>());
      SmallVector<APInt> left, right, masks;
      bool legal = true;
      for (int64_t first = 0; first < type.getNumElements(); first += 4) {
        uint64_t source = starts[first].getLimitedValue(), destination = ends[first].getLimitedValue();
        if (source > 8 - width || destination > 8 - width) { legal = false; break; }
        for (int64_t lane = first + 1; lane < first + 4; ++lane)
          if (starts[lane] != starts[first] || ends[lane] != ends[first]) legal = false;
        if (!legal) break;
        left.emplace_back(32, destination > source ? destination - source : 0);
        right.emplace_back(32, source > destination ? source - destination : 0);
        uint32_t mask = ((1u << width) - 1) << destination;
        masks.emplace_back(32, mask * 0x01010101u);
      }
      if (!legal) continue;
      // Four equal byte-field shifts share a native word shift. Masking the
      // destination bits removes every cross-byte contribution. In particular,
      // numerical one-bit fields need no predicate-register round trip.
      b.setInsertionPoint(shift);
      auto wordType = VectorType::get({type.getNumElements() / 4}, b.getI32Type());
      auto constant = [&](ArrayRef<APInt> values) -> Value {
        return arith::ConstantOp::create(b, shift.getLoc(), wordType,
            DenseIntElementsAttr::get(wordType, values));
      };
      Value words = vector::BitCastOp::create(b, shift.getLoc(), wordType, bytes.getIn());
      words = arith::ShLIOp::create(b, shift.getLoc(), words, constant(left));
      words = arith::ShRUIOp::create(b, shift.getLoc(), words, constant(right));
      words = arith::AndIOp::create(b, shift.getLoc(), words, constant(masks));
      b.replaceOpWithNewOp<vector::BitCastOp>(shift, type, words);
    }
  }
public:
  explicit LegalizeSubbyteProjections(bool selectedConsumersOnly = false) : selectedConsumersOnly(selectedConsumersOnly) {}
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeSubbyteProjections)
  StringRef getArgument() const override { return "weft-x86-legalize-subbyte-projections"; }
  StringRef getDescription() const override {
    return "Express widened subbyte permutations using legal integer vector lanes";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto lookup = selectedConsumersOnly ? nativeIntegerClosure(getOperation()) : llvm::DenseSet<Operation *>();
    SmallVector<Operation *> extensions;
    getOperation().walk([&](Operation *op) {
      if (isa<arith::ExtUIOp, arith::ExtSIOp>(op) && (!selectedConsumersOnly || lookup.contains(op))) extensions.push_back(op);
    });
    IRRewriter b(&getContext());
    for (auto *extension : extensions) {
      auto shuffle = extension->getOperand(0).getDefiningOp<vector::ShuffleOp>();
      if (!shuffle) continue;
      auto result = dyn_cast<VectorType>(extension->getResult(0).getType());
      auto input = cast<VectorType>(shuffle.getType());
      if (!result || result.isScalable() || input.getElementTypeBitWidth() >= 8 ||
          input.getElementTypeBitWidth() == 1 || result.getElementTypeBitWidth() < 8) continue;
      b.setInsertionPoint(extension);
      auto extend = [&](Value value) -> Value {
        Type type = cast<VectorType>(value.getType()).clone(result.getElementType());
        if (isa<arith::ExtUIOp>(extension))
          return arith::ExtUIOp::create(b, extension->getLoc(), type, value);
        return arith::ExtSIOp::create(b, extension->getLoc(), type, value);
      };
      // Extension commutes with a pure lane permutation. Preserve its signed
      // interpretation, but avoid legalizing an i2/i4 shuffle as packed bits.
      Value left = extend(shuffle.getV1()), right = extend(shuffle.getV2());
      Value value = vector::ShuffleOp::create(b, extension->getLoc(), left, right, shuffle.getMask());
      b.replaceOp(extension, value);
    }
    materializeBytePlanes(b);
  }
};

// Consumers of a decoded record often request small, disjoint lane packets.
// Decode those packets from the same saved bytes instead of decoding a wide
// row-major vector and transposing it again for each consumer. Only integer
// lane-local expressions move; reads, reduction boundaries and FP rounding stay.
bool integerLaneLocal(Operation *op, bool byteMultiply) {
    if (auto join = dyn_cast_or_null<arith::OrIOp>(op)) {
      auto type = dyn_cast<VectorType>(join.getType());
      auto byteExtension = [](Value value) {
        if (auto mask = value.getDefiningOp<arith::AndIOp>()) {
          if (mask.getLhs().getDefiningOp<arith::ConstantOp>()) value = mask.getRhs();
          else if (mask.getRhs().getDefiningOp<arith::ConstantOp>()) value = mask.getLhs();
        }
        auto extend = value.getDefiningOp<arith::ExtUIOp>();
        return extend && getElementTypeOrSelf(extend.getIn().getType()).isInteger(8);
      };
      if (type && llvm::isPowerOf2_32(type.getElementTypeBitWidth()) &&
          type.getElementTypeBitWidth() >= 16)
        for (unsigned side = 0; side != 2; ++side) {
          auto shift = join->getOperand(side).getDefiningOp<arith::ShLIOp>();
          APInt amount;
          if (shift && matchPattern(shift.getRhs(), m_ConstantInt(&amount)) &&
              amount.getLimitedValue() >= 8 && amount.getLimitedValue() < type.getElementTypeBitWidth() &&
              amount.getLimitedValue() % 8 == 0 && byteExtension(shift.getLhs()) &&
              byteExtension(join->getOperand(1 - side)))
            // Keep native word assembly intact. Projecting sparse consumers
            // through both byte planes fragments a single word permutation
            // into byte extraction, narrowing and reconstruction sequences.
            return false;
        }
    }
    // Native narrow products can follow their disjoint consumer packets too.
    // The caller proves the complete demand set and shares identical packets,
    // so variable scales do not require materializing the full wide product.
    if (auto multiply = dyn_cast_or_null<arith::MulIOp>(op)) {
      if (!byteMultiply) return false;
      auto result = dyn_cast<IntegerType>(getElementTypeOrSelf(multiply.getType()));
      if (result && result.getWidth() <= 16) return true;
      for (unsigned side = 0; side != 2; ++side) {
        if (!multiply->getOperand(1 - side).getDefiningOp<arith::ConstantOp>()) continue;
        Value input = multiply->getOperand(side);
        auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(input.getType()));
        if (type && type.getWidth() <= 8) return true;
        if (Operation *extend = input.getDefiningOp();
            isa_and_nonnull<arith::ExtUIOp, arith::ExtSIOp>(extend) &&
            cast<IntegerType>(getElementTypeOrSelf(extend->getOperand(0).getType())).getWidth() <= 8) return true;
      }
      return false;
    }
    return isa_and_nonnull<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp,
        arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::ShLIOp,
        arith::ShRUIOp, arith::ShRSIOp, arith::AddIOp, arith::SubIOp,
        arith::CmpIOp, arith::SelectOp>(op);
}

class ProjectIntegerExpressions
    : public PassWrapper<ProjectIntegerExpressions, OperationPass<ModuleOp>> {
  bool selectedConsumersOnly;
  bool uniformWordShift(Value value, ArrayRef<int64_t> indices, unsigned depth = 0) {
    Operation *op = value.getDefiningOp();
    auto type = dyn_cast<VectorType>(value.getType());
    if (!op || !type || indices.empty() || depth > 8) return false;
    if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(op)) {
      DenseIntElementsAttr shifts;
      if (matchPattern(op->getOperand(1), m_Constant(&shifts)) && !shifts.isSplat()) {
        SmallVector<APInt> values(shifts.getValues<APInt>());
        if (llvm::all_of(indices, [&](int64_t index) { return values[index] == values[indices.front()]; }))
          return true;
      }
    }
    if (!integerLaneLocal(op, true)) return false;
    return llvm::any_of(op->getOperands(), [&](Value operand) {
      auto input = dyn_cast<VectorType>(operand.getType());
      return input && input.getShape() == type.getShape() && uniformWordShift(operand, indices, depth + 1);
    });
  }
  bool projectionScope(Value value, Operation *root) {
    Operation *definition = value.getDefiningOp();
    if (!definition) return false;
    Operation *edge = root;
    while (edge->getBlock() != definition->getBlock()) {
      edge = edge->getParentOp();
      if (!isa_and_nonnull<scf::IfOp>(edge)) return false;
    }
    if (definition == root || value.hasOneUse()) return true;
    if (edge == root) return false;
    // The old representation is consumed only by alternative implementations
    // of this conditional. Its remaining pure producers can sink into the
    // other arm after projection; there is no extra work on the selected path.
    return llvm::all_of(value.getUsers(), [&](Operation *user) {
      while (user && user->getBlock() != definition->getBlock()) user = user->getParentOp();
      return user == edge;
    });
  }
  Value project(IRRewriter &b, Location loc, Value value, ArrayRef<int64_t> indices,
                Operation *root, unsigned depth = 0) {
    auto type = cast<VectorType>(value.getType());
    auto resultType = type.cloneWith({static_cast<int64_t>(indices.size())}, type.getElementType());
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto elements = dyn_cast<DenseIntElementsAttr>(constant.getValue())) {
        SmallVector<APInt> source(elements.getValues<APInt>()), selected;
        for (int64_t index : indices) selected.push_back(source[index]);
        return arith::ConstantOp::create(b, loc, resultType, DenseIntElementsAttr::get(resultType, selected));
      }
    }
    if (depth < 32) {
      if (auto bits = value.getDefiningOp<vector::BitCastOp>()) {
        auto inputType = cast<VectorType>(bits.getSource().getType());
        unsigned sourceBits = inputType.getElementTypeBitWidth(), outputBits = type.getElementTypeBitWidth();
        if (projectionScope(value, root) &&
            isa<IntegerType>(inputType.getElementType()) && isa<IntegerType>(type.getElementType()) &&
            sourceBits >= 8 && outputBits > sourceBits && outputBits % sourceBits == 0) {
          int64_t group = outputBits / sourceBits;
          SmallVector<int64_t> elements;
          for (int64_t index : indices)
            for (int64_t part = 0; part < group; ++part) elements.push_back(index * group + part);
          Value input = project(b, loc, bits.getSource(), elements, root, depth + 1);
          return vector::BitCastOp::create(b, loc, resultType, input);
        }
        if (projectionScope(value, root) &&
            isa<IntegerType>(inputType.getElementType()) && isa<IntegerType>(type.getElementType()) &&
            outputBits >= 8 && sourceBits > outputBits && sourceBits % outputBits == 0) {
          int64_t group = sourceBits / outputBits;
          SmallVector<int64_t> words;
          bool complete = indices.size() % group == 0;
          for (size_t first = 0; complete && first < indices.size(); first += group) {
            if (indices[first] % group) { complete = false; break; }
            for (int64_t part = 1; part < group; ++part)
              complete &= indices[first + part] == indices[first] + part;
            words.push_back(indices[first] / group);
          }
          // A consumer selecting complete words can project their producers
          // before splitting them into bytes. Constant field shifts then stay
          // local to each demanded packet instead of decoding every plane and
          // permuting a wider word vector back into the same bytes.
          if (complete && uniformWordShift(bits.getSource(), words)) {
            Value input = project(b, loc, bits.getSource(), words, root, depth + 1);
            return vector::BitCastOp::create(b, loc, resultType, input);
          }
        }
      }
      if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
        int64_t size = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
        SmallVector<int64_t> composed;
        Value source;
        for (int64_t index : indices) {
          int64_t selected = shuffle.getMask()[index];
          if (selected < 0) break;
          Value operand = selected < size ? shuffle.getV1() : shuffle.getV2();
          if (source && source != operand) break;
          source = operand;
          composed.push_back(selected < size ? selected : selected - size);
        }
        if (composed.size() == indices.size()) {
          auto sourceType = cast<VectorType>(source.getType());
          auto function = root->getParentOfType<func::FuncOp>();
          auto bits = function ? function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits") : IntegerAttr();
          // A broadcast can make disjoint consumer packets demand the same
          // small producer repeatedly. Lane-count conservation at the root
          // does not bound this inner work. Keep a native-register producer
          // shared rather than redoing its shifts, signs and scales per use.
          if (bits && sourceType.getNumElements() < type.getNumElements() &&
              sourceType.getNumElements() * sourceType.getElementTypeBitWidth() <= bits.getInt()) {
            llvm::SmallDenseSet<int64_t> distinct;
            for (int64_t index : composed) distinct.insert(index);
            if (distinct.size() < composed.size() &&
                integerLaneLocal(source.getDefiningOp(), selectedConsumersOnly))
              return vector::ShuffleOp::create(b, loc, source, source, composed);
          }
          return project(b, loc, source, composed, root, depth + 1);
        }
      }
      Operation *definition = value.getDefiningOp();
      if (integerLaneLocal(definition, selectedConsumersOnly) && projectionScope(value, root) &&
          llvm::all_of(definition->getOperands(), [&](Value operand) {
            auto vector = dyn_cast<VectorType>(operand.getType());
            return vector && vector.getShape() == type.getShape() && !vector.isScalable();
          })) {
        SmallVector<Value> operands;
        for (Value operand : definition->getOperands())
          operands.push_back(project(b, loc, operand, indices, root, depth + 1));
        OperationState state(loc, definition->getName());
        state.addOperands(operands); state.addTypes(resultType); state.addAttributes(definition->getAttrs());
        return b.create(state)->getResult(0);
      }
    }
    return vector::ShuffleOp::create(b, loc, value, value, indices);
  }
  void conversionPackets(IRRewriter &b) {
    SmallVector<Operation *> conversions;
    getOperation().walk([&](Operation *op) {
      if (isa<arith::SIToFPOp, arith::UIToFPOp>(op)) conversions.push_back(op);
    });
    for (Operation *op : conversions) {
      auto type = dyn_cast<VectorType>(op->getResult(0).getType());
      auto source = dyn_cast<VectorType>(op->getOperand(0).getType());
      auto function = op->getParentOfType<func::FuncOp>();
      auto bits = function ? function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits") : IntegerAttr();
      if (!type || !source || !bits || type.isScalable() || type.getRank() != 1 ||
          !type.getElementType().isF32() || source.getElementTypeBitWidth() > 32 ||
          !op->getOperand(0).hasOneUse()) continue;
      // Preserve a native integer packet before widening. Splitting byte
      // producers into F32-sized packets multiplies shifts, masks and adds
      // fourfold even though the conversion itself still needs the same number
      // of output registers. LLVM can split the wider conversion independently.
      unsigned width = source.getElementTypeBitWidth();
      int64_t lanes = bits.getInt() / std::max(8u, width);
      int64_t count = type.getNumElements();
      // Subbyte values still need native-width legalization. Small producers
      // also benefit from exposing each constant bit plane before conversion;
      // widening and permuting a mixed plane costs more than two direct casts.
      if (width < 8 || count <= lanes) lanes = bits.getInt() / 32;
      if (count <= lanes) continue;
      b.setInsertionPoint(op);
      Value result = arith::ConstantOp::create(b, op->getLoc(), type,
          DenseElementsAttr::get(type, b.getF32FloatAttr(0)));
      // A conversion consumes native output packets. Project the same disjoint
      // integer lanes before widening, so constant bit planes and byte shuffles
      // do not become a wide variable-shift program. Reads stay at their source
      // snapshot and shared integer producers are not recomputed.
      for (int64_t first = 0; first < count; first += lanes) {
        int64_t size = std::min(lanes, count - first);
        SmallVector<int64_t> indices;
        for (int64_t lane = first; lane < first + size; ++lane) indices.push_back(lane);
        Value input = project(b, op->getLoc(), op->getOperand(0), indices, op);
        auto packetType = VectorType::get({size}, type.getElementType());
        OperationState state(op->getLoc(), op->getName());
        state.addOperands(input); state.addTypes(packetType); state.addAttributes(op->getAttrs());
        Value converted = b.create(state)->getResult(0);
        result = vector::InsertStridedSliceOp::create(b, op->getLoc(), converted, result,
            ArrayRef<int64_t>{first}, ArrayRef<int64_t>{1});
      }
      b.replaceOp(op, result);
    }
  }
public:
  explicit ProjectIntegerExpressions(bool selectedConsumersOnly = false) : selectedConsumersOnly(selectedConsumersOnly) {}
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProjectIntegerExpressions)
  StringRef getArgument() const override { return "weft-x86-project-integer-expressions"; }
  StringRef getDescription() const override {
    return "Project disjoint consumer packets through integer expressions without repeating arithmetic or reads";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    if (!selectedConsumersOnly) conversionPackets(b);
    auto lookup = selectedConsumersOnly ? nativeIntegerClosure(getOperation()) : llvm::DenseSet<Operation *>();
    SmallVector<Operation *> roots;
    getOperation().walk([&](Operation *op) {
      if ((integerLaneLocal(op, selectedConsumersOnly) || isa<vector::BitCastOp>(op)) &&
          op->getNumResults() == 1 && (!selectedConsumersOnly || lookup.contains(op))) roots.push_back(op);
    });
    for (Operation *root : llvm::reverse(roots)) {
      auto type = dyn_cast<VectorType>(root->getResult(0).getType());
      if (!type || type.isScalable() || type.getRank() != 1 || root->use_empty()) continue;
      llvm::SetVector<Operation *> consumers;
      for (Operation *user : root->getUsers()) consumers.insert(user);
      SmallVector<SmallVector<int64_t>> unique, reverse;
      SmallVector<unsigned> packetForConsumer;
      int64_t total = 0;
      bool eligible = true;
      for (Operation *consumer : consumers) {
        auto shuffle = dyn_cast<vector::ShuffleOp>(consumer);
        if (!shuffle || shuffle->getBlock() != root->getBlock()) { eligible = false; break; }
        int64_t firstSize = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
        llvm::SetVector<int64_t> demand;
        SmallVector<int64_t> expand;
        for (int64_t index : shuffle.getMask()) {
          if (index < 0 || (index < firstSize ? shuffle.getV1() : shuffle.getV2()) != root->getResult(0)) {
            eligible = false; break;
          }
          if (index >= firstSize) index -= firstSize;
          demand.insert(index);
          expand.push_back(index);
        }
        SmallVector<int64_t> packet(demand.begin(), demand.end());
        llvm::sort(packet);
        for (int64_t &index : expand) index = llvm::lower_bound(packet, index) - packet.begin();
        auto existing = llvm::find(unique, packet);
        if (existing == unique.end()) {
          packetForConsumer.push_back(unique.size());
          total += packet.size();
          unique.push_back(std::move(packet));
        } else packetForConsumer.push_back(existing - unique.begin());
        reverse.push_back(std::move(expand));
      }
      // Total decoded lanes cannot increase, and every packet must shrink.
      if (!eligible || total > type.getNumElements() || llvm::any_of(unique, [&](auto &demand) {
            return demand.size() >= static_cast<size_t>(type.getNumElements());
          })) continue;
      auto function = root->getParentOfType<func::FuncOp>();
      auto bits = function ? function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits") : IntegerAttr();
      if (bits && selectedConsumersOnly) {
        int64_t width = std::max<int64_t>(8, type.getElementTypeBitWidth());
        auto registers = [&](int64_t lanes) {
          return llvm::divideCeil(lanes * width, bits.getInt());
        };
        int64_t projected = 0;
        for (const auto &demand : unique) projected += registers(demand.size());
        // Equal logical lane counts do not imply equal physical work: several
        // short packets can each require a native instruction where the shared
        // producer needs just one. Preserve that producer and its permutations.
        // Separating mixed constant bit planes can replace variable shifts
        // with immediate shifts, so these packets have a distinct native cost.
        if (projected > registers(type.getNumElements()) &&
            !llvm::all_of(unique, [&](const auto &demand) {
              return uniformWordShift(root->getResult(0), demand);
            })) continue;
      }
      // Identical lane demands share one decoded packet even when their uses
      // broadcast or order those lanes differently. Charge distinct producers,
      // not their number of consumers, and keep all reads at their snapshots.
      SmallVector<Value> packets;
      for (auto [packet, demand] : llvm::enumerate(unique)) {
        Operation *first = nullptr;
        for (auto [index, consumer] : llvm::enumerate(consumers))
          if (packetForConsumer[index] == packet && (!first || consumer->isBeforeInBlock(first))) first = consumer;
        // Pure packets are needed at their first consumer, not simultaneously
        // at the old wide definition. Source reads remain saved SSA values.
        // Keep this issue order visible to LLVM without extending every
        // decoded packet's live interval across unrelated consumers.
        b.setInsertionPoint(first);
        packets.push_back(project(b, root->getLoc(), root->getResult(0), demand, root));
      }
      for (auto [index, consumer] : llvm::enumerate(consumers)) {
        b.setInsertionPoint(consumer);
        Value selected = packets[packetForConsumer[index]];
        Value result = vector::ShuffleOp::create(b, consumer->getLoc(), selected, selected, reverse[index]);
        b.replaceOp(consumer, result);
      }
    }
    SmallVector<scf::IfOp> choices;
    getOperation().walk([&](scf::IfOp choice) { choices.push_back(choice); });
    DominanceInfo dominance(getOperation());
    for (auto choice : choices)
      controlFlowSink(choice->getRegions(), dominance,
          [&](Operation *op, Region *) {
            return (integerLaneLocal(op, true) || isa<vector::ShuffleOp, vector::BitCastOp>(op)) &&
                llvm::all_of(op->getResultTypes(), [](Type type) {
                  return getElementTypeOrSelf(type).isIntOrIndex();
                });
          }, [&](Operation *op, Region *region) { op->moveBefore(&region->front(), region->front().begin()); });
  }
};

class MergeIntegerPackets : public PassWrapper<MergeIntegerPackets, OperationPass<ModuleOp>> {
  Value concatenate(IRRewriter &b, Location loc, ArrayRef<Value> parts) {
    auto element = cast<VectorType>(parts.front().getType()).getElementType();
    int64_t size = 0;
    for (Value part : parts) size += cast<VectorType>(part.getType()).getNumElements();
    auto resultType = VectorType::get({size}, element);
    SmallVector<APInt> literals;
    for (Value part : parts) {
      auto constant = part.getDefiningOp<arith::ConstantOp>();
      auto data = constant ? dyn_cast<DenseIntElementsAttr>(constant.getValue()) : DenseIntElementsAttr();
      if (!data) break;
      llvm::append_range(literals, data.getValues<APInt>());
    }
    if (literals.size() == static_cast<size_t>(size))
      return arith::ConstantOp::create(b, loc, resultType, DenseIntElementsAttr::get(resultType, literals));
    Value source;
    SmallVector<int64_t> mask;
    for (Value part : parts) {
      auto shuffle = part.getDefiningOp<vector::ShuffleOp>();
      if (!shuffle) break;
      int64_t first = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
      for (int64_t position : shuffle.getMask()) {
        if (position < 0) break;
        Value input = position < first ? shuffle.getV1() : shuffle.getV2();
        if (source && source != input) break;
        source = input;
        mask.push_back(position < first ? position : position - first);
      }
    }
    if (mask.size() == static_cast<size_t>(size))
      return vector::ShuffleOp::create(b, loc, source, source, mask);
    Value result = arith::ConstantOp::create(b, loc, resultType,
        DenseElementsAttr::get(resultType, b.getIntegerAttr(element, 0)));
    int64_t offset = 0;
    for (Value part : parts) {
      result = vector::InsertStridedSliceOp::create(b, loc, part, result,
          ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{1});
      offset += cast<VectorType>(part.getType()).getNumElements();
    }
    return result;
  }
  bool mergeable(ArrayRef<Value> parts, Block *block, int64_t size, unsigned vectorBits) {
    Operation *first = parts.front().getDefiningOp();
    if (!integerLaneLocal(first) || first->getNumResults() != 1) return false;
    for (Value part : parts) {
      Operation *op = part.getDefiningOp();
      if (!op || op->getName() != first->getName() || op->getAttrs() != first->getAttrs() ||
          op->getBlock() != block || !part.hasOneUse() || op->getNumOperands() != first->getNumOperands()) return false;
      auto output = cast<VectorType>(part.getType());
      if (size * output.getElementTypeBitWidth() > 2 * vectorBits) return false;
      for (auto [a, c] : llvm::zip(op->getOperands(), first->getOperands())) {
        auto type = dyn_cast<VectorType>(a.getType());
        auto other = dyn_cast<VectorType>(c.getType());
        if (!type || !other || type.getRank() != 1 || type.isScalable() ||
            type.getElementType() != other.getElementType() ||
            type.getNumElements() != output.getNumElements() ||
            size * type.getElementTypeBitWidth() > 2 * vectorBits) return false;
      }
    }
    return true;
  }
  Value merge(IRRewriter &b, Location loc, ArrayRef<Value> parts, Block *block,
              int64_t size, unsigned vectorBits, unsigned depth = 0) {
    if (depth == 16 || !mergeable(parts, block, size, vectorBits)) return concatenate(b, loc, parts);
    Operation *first = parts.front().getDefiningOp();
    SmallVector<Value> operands;
    for (unsigned i = 0; i < first->getNumOperands(); ++i) {
      SmallVector<Value> inputs;
      for (Value part : parts) inputs.push_back(part.getDefiningOp()->getOperand(i));
      operands.push_back(merge(b, loc, inputs, block, size, vectorBits, depth + 1));
    }
    auto type = VectorType::get({size}, cast<VectorType>(parts.front().getType()).getElementType());
    OperationState state(loc, first->getName());
    state.addOperands(operands); state.addTypes(type); state.addAttributes(first->getAttrs());
    return b.create(state)->getResult(0);
  }
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeIntegerPackets)
  StringRef getArgument() const override { return "weft-x86-merge-integer-packets"; }
  StringRef getDescription() const override { return "Coalesce disjoint integer fragments within one native consumer packet"; }
  void runOnOperation() override {
    SmallVector<vector::InsertStridedSliceOp> roots;
    getOperation().walk([&](vector::InsertStridedSliceOp op) {
      if (op->use_empty()) return;
      if (op->hasOneUse() && isa<vector::InsertStridedSliceOp>(*op->getUsers().begin())) return;
      // This pass finishes a selected register lookup's index packet. Other
      // materializations keep their existing scheduling and resource choices.
      if (!llvm::all_of(op->getUsers(), [&](Operation *user) {
            auto call = dyn_cast<func::CallOp>(user);
            return call && (call.getCallee() == "llvm.x86.avx2.pshuf.b" ||
                            call.getCallee() == "llvm.x86.avx512.pshuf.b.512") &&
                   call.getOperand(1) == op.getResult();
          })) return;
      roots.push_back(op);
    });
    IRRewriter b(&getContext());
    for (auto root : roots) {
      auto type = cast<VectorType>(root.getType());
      auto function = root->getParentOfType<func::FuncOp>();
      auto bits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits");
      if (!bits || type.getRank() != 1 || type.isScalable() || !isa<IntegerType>(type.getElementType()) ||
          type.getNumElements() * type.getElementTypeBitWidth() > bits.getInt()) continue;
      int64_t remaining = type.getNumElements();
      SmallVector<Value> parts;
      Value current = root.getResult();
      while (auto insert = current.getDefiningOp<vector::InsertStridedSliceOp>()) {
        if (insert != root && !current.hasOneUse()) break;
        if (insert.getOffsets().size() != 1 || cast<IntegerAttr>(insert.getStrides()[0]).getInt() != 1) break;
        int64_t count = cast<VectorType>(insert.getValueToStore().getType()).getNumElements();
        int64_t offset = cast<IntegerAttr>(insert.getOffsets()[0]).getInt();
        if (offset < 0 || offset + count != remaining) break;
        parts.push_back(insert.getValueToStore());
        remaining = offset; current = insert.getDest();
        if (!remaining) break;
      }
      if (remaining || parts.size() < 2) continue;
      std::reverse(parts.begin(), parts.end());
      if (!mergeable(parts, root->getBlock(), type.getNumElements(), bits.getInt())) continue;
      b.setInsertionPoint(root);
      Value result = merge(b, root.getLoc(), parts, root->getBlock(), type.getNumElements(), bits.getInt());
      b.replaceOp(root, result);
    }
  }
};

// SelectionDAG's cross-block register breakdown scalarizes non-power-of-two
// vectors even when their arithmetic is vectorized. Keep their exact lane
// sequence in padded vectors across CFG edges and their elementwise updates.
// Splitting only the edge is insufficient: InstCombine reconstructs the original
// non-power-of-two PHI when its update still uses that type. This changes neither
// source accumulators nor numerical/reduction boundaries.
class LegalizeVectorCarries
    : public PassWrapper<LegalizeVectorCarries, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeVectorCarries)
  StringRef getArgument() const override { return "weft-x86-legalize-vector-carries"; }
  StringRef getDescription() const override {
    return "Keep non-power-of-two vector carries and updates in padded SIMD carriers";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isDeclaration()) continue;
      DenseMap<Block *, SmallVector<Type>> plans;
      DenseMap<Value, Value> padded;
      auto wideType = [&](Type type) -> Type {
        auto vector = dyn_cast<VectorType>(type);
        if (!vector || vector.getRank() != 1 || vector.isScalable() ||
            llvm::isPowerOf2_64(vector.getNumElements())) return type;
        return VectorType::get({int64_t(llvm::PowerOf2Ceil(vector.getNumElements()))}, vector.getElementType());
      };
      auto unpack = [&](Value value, Type original, Location loc) -> Value {
        if (value.getType() == original) return value;
        int64_t count = cast<VectorType>(original).getNumElements();
        Value narrow = vector::ExtractStridedSliceOp::create(b, loc, value,
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{count}, ArrayRef<int64_t>{1});
        padded[narrow] = value;
        return narrow;
      };
      auto pack = [&](Value value) -> Value {
        if (auto found = padded.find(value); found != padded.end()) return found->second;
        Type type = wideType(value.getType());
        if (type == value.getType()) return value;
        OpBuilder::InsertionGuard guard(b);
        if (auto argument = dyn_cast<BlockArgument>(value)) b.setInsertionPointToStart(argument.getOwner());
        else b.setInsertionPointAfter(value.getDefiningOp());
        Value zero = arith::ConstantOp::create(b, value.getLoc(), type, b.getZeroAttr(type));
        Value result = vector::InsertStridedSliceOp::create(b, value.getLoc(), value, zero,
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        padded[value] = result;
        return result;
      };
      for (Block &block : llvm::drop_begin(function.getBody())) {
        SmallVector<Type> plan;
        bool changed = false;
        for (BlockArgument argument : block.getArguments()) {
          Type type = wideType(argument.getType());
          changed |= type != argument.getType();
          plan.push_back(type);
        }
        if (changed) plans[&block] = std::move(plan);
      }
      if (plans.empty()) continue;
      for (auto &[block, plan] : plans) {
        SmallVector<BlockArgument> original(block->getArguments());
        b.setInsertionPointToStart(block);
        for (auto [argument, type] : llvm::zip(original, plan)) {
          Value replacement = block->addArgument(type, argument.getLoc());
          argument.replaceAllUsesWith(unpack(replacement, argument.getType(), argument.getLoc()));
        }
        for (BlockArgument argument : llvm::reverse(original)) block->eraseArgument(argument.getArgNumber());
      }
      SmallVector<Operation *> updates;
      function.walk([&](Operation *op) {
        // Padding must not introduce division by zero, new memory effects or a
        // reduction term. These operations act independently on each lane and
        // zero padding is safe for their numerical contract.
        if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, math::FmaOp,
                arith::AddIOp, arith::SubIOp, arith::MulIOp,
                arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
                arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                arith::ExtFOp, arith::TruncFOp, arith::BitcastOp>(op) &&
            op->getNumResults() == 1 && wideType(op->getResult(0).getType()) != op->getResult(0).getType())
          updates.push_back(op);
      });
      for (Operation *op : updates) {
        b.setInsertionPoint(op);
        SmallVector<Value> operands;
        for (Value value : op->getOperands()) operands.push_back(pack(value));
        OperationState state(op->getLoc(), op->getName());
        state.addOperands(operands);
        state.addTypes(wideType(op->getResult(0).getType()));
        state.addAttributes(op->getAttrs());
        Value replacement = b.create(state)->getResult(0);
        padded.erase(op->getResult(0));
        b.replaceOp(op, unpack(replacement, op->getResult(0).getType(), op->getLoc()));
      }
      // Rewrite edges to the same padded representation. Both destinations of a
      // conditional branch may be the same block and need independent operands.
      for (Block &block : function.getBody()) {
        Operation *terminator = block.getTerminator();
        auto branch = dyn_cast<BranchOpInterface>(terminator);
        if (!branch) continue;
        for (unsigned edge = 0; edge < terminator->getNumSuccessors(); ++edge) {
          auto found = plans.find(terminator->getSuccessor(edge));
          if (found == plans.end()) continue;
          auto operands = branch.getSuccessorOperands(edge);
          if (operands.getProducedOperandCount()) {
            terminator->emitError("x86 vector carry requires explicit branch operands");
            return signalPassFailure();
          }
          b.setInsertionPoint(terminator);
          SmallVector<Value> expanded;
          for (Value value : operands.getForwardedOperands()) expanded.push_back(pack(value));
          operands.getMutableForwardedOperands().assign(expanded);
        }
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86VectorProjectionPass() {
  return std::make_unique<CoalesceVectorProjections>();
}

std::unique_ptr<Pass> weft::createX86SubbyteProjectionPass(bool selectedConsumersOnly) {
  return std::make_unique<LegalizeSubbyteProjections>(selectedConsumersOnly);
}

std::unique_ptr<Pass> weft::createX86IntegerProjectionPass(bool selectedConsumersOnly) {
  return std::make_unique<ProjectIntegerExpressions>(selectedConsumersOnly);
}

std::unique_ptr<Pass> weft::createX86IntegerPacketMergePass() {
  return std::make_unique<MergeIntegerPackets>();
}

std::unique_ptr<Pass> weft::createX86VectorCarryPass() {
  return std::make_unique<LegalizeVectorCarries>();
}
