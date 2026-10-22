#include "Weft/Target/X86Compiler.h"
#include "IndexedValues.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <map>
#include <optional>

using namespace mlir;
namespace {
// Prove repeated predicate lanes by their SSA expression, then save only one
// representative per group. Do not re-evaluate the producer at an indexed use:
// that can turn a cheap snapshot into repeated arithmetic in a hot loop.
class PredicateLanes {
  DenseMap<std::pair<Value, int64_t>, unsigned> known;
  std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> expressions;
  unsigned next = 2; // IDs zero and one are literal predicates.
public:
  unsigned identify(Value value, int64_t lane, unsigned depth = 0) {
    if (!isa<VectorType>(value.getType())) lane = -1;
    auto key = std::pair{value, lane};
    if (auto found = known.find(key); found != known.end()) return found->second;
    unsigned id = next++;
    if (depth < 32) {
      auto recurse = [&](Value source, int64_t index) { return identify(source, index, depth + 1); };
      if (auto c = value.getDefiningOp<arith::ConstantOp>()) {
        if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue()); dense && lane >= 0)
          id = dense.getValues<APInt>()[lane].getBoolValue();
        else if (auto bit = dyn_cast<IntegerAttr>(c.getValue())) id = bit.getValue().getBoolValue();
      } else if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
        int64_t index = shuffle.getMask()[lane], split = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
        if (index >= 0) id = index < split ? recurse(shuffle.getV1(), index) : recurse(shuffle.getV2(), index - split);
      } else if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>()) {
        if (!isa<VectorType>(broadcast.getSource().getType())) id = recurse(broadcast.getSource(), -1);
      } else if (auto pack = value.getDefiningOp<vector::FromElementsOp>()) {
        id = recurse(pack.getOperand(lane), -1);
      } else if (auto insert = value.getDefiningOp<LLVM::InsertElementOp>()) {
        Value cursor = value;
        while (auto element = cursor.getDefiningOp<LLVM::InsertElementOp>()) {
          APInt position;
          if (!matchPattern(element.getPosition(), m_ConstantInt(&position))) break;
          if (lane == position.getSExtValue()) { cursor = element.getValue(); break; }
          cursor = element.getVector();
        }
        if (cursor != value) id = recurse(cursor, lane);
      } else if (Operation *op = value.getDefiningOp(); op && isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
        unsigned a = recurse(op->getOperand(0), lane), c = recurse(op->getOperand(1), lane);
        if (a > c) std::swap(a, c);
        unsigned kind = isa<arith::AndIOp>(op) ? 0 : isa<arith::OrIOp>(op) ? 1 : 2;
        if (a == c) id = kind == 2 ? 0 : a;
        else if (kind == 0 && a == 0) id = 0;
        else if (kind == 0 && a == 1) id = c;
        else if ((kind == 1 || kind == 2) && a == 0) id = c;
        else if (kind == 1 && a == 1) id = 1;
        else {
          auto [found, inserted] = expressions.emplace(std::tuple{kind, a, c}, id);
          id = found->second;
        }
      }
    }
    known[key] = id;
    return id;
  }

  SmallVector<int64_t> representatives(Value source, int64_t &repeat) {
    int64_t count = cast<VectorType>(source.getType()).getNumElements();
    SmallVector<unsigned> lanes;
    for (int64_t lane = 0; lane < count; ++lane) lanes.push_back(identify(source, lane));
    repeat = 1;
    while (repeat < count && lanes[repeat] == lanes.front()) ++repeat;
    if (count % repeat) return {};
    int64_t period = count / repeat;
    for (int64_t i = repeat; i < count; i += repeat)
      if (lanes[i] == lanes.front()) { period = i / repeat; break; }
    if (period * 2 > count) return {};
    for (int64_t i = 0; i < count; ++i)
      if (lanes[i] != lanes[(i / repeat % period) * repeat]) return {};
    SmallVector<int64_t> result;
    for (int64_t i = 0; i < period; ++i) result.push_back(i * repeat);
    return result;
  }
};

unsigned indexWidth(Type type) { return isa<IndexType>(type) ? 64 : cast<IntegerType>(type).getWidth(); }

unsigned trailingZeros(Value value, unsigned depth = 0) {
  if (depth > 12) return 0;
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) return constant.countr_zero();
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    if (indexWidth(cast.getIn().getType()) == indexWidth(value.getType())) return trailingZeros(cast.getIn(), depth + 1);
  if (auto mul = value.getDefiningOp<arith::MulIOp>())
    return std::min(indexWidth(value.getType()), trailingZeros(mul.getLhs(), depth + 1) + trailingZeros(mul.getRhs(), depth + 1));
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return std::min(trailingZeros(add.getLhs(), depth + 1), trailingZeros(add.getRhs(), depth + 1));
  return 0;
}

// Exact low-bit projection, including pointer-width wrap. In particular,
// ((x * 4 + byte) / 4) % rows has the same predicate index for every byte in
// the word. Keep just the selected bits, rather than asserting x cannot wrap.
Value indexBits(Value value, unsigned shift, unsigned width, OpBuilder &b, Location loc, unsigned depth = 0) {
  auto number = [&](uint64_t n) -> Value {
    return arith::ConstantOp::create(b, loc, b.getIntegerAttr(value.getType(), n));
  };
  uint64_t mask = (uint64_t(1) << width) - 1;
  APInt c;
  if (matchPattern(value, m_ConstantInt(&c))) return number(c.lshr(shift).getZExtValue() & mask);
  if (depth < 12) {
    auto recurse = [&](Value v, unsigned s) { return indexBits(v, s, width, b, loc, depth + 1); };
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      if (indexWidth(cast.getIn().getType()) == indexWidth(value.getType())) {
        Value input = recurse(cast.getIn(), shift);
        return arith::IndexCastOp::create(b, loc, value.getType(), input);
      }
    }
    if (auto select = value.getDefiningOp<arith::SelectOp>())
      return arith::SelectOp::create(b, loc, select.getCondition(),
          recurse(select.getTrueValue(), shift), recurse(select.getFalseValue(), shift));
    if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
      Value input = mul.getLhs();
      bool constant = matchPattern(mul.getRhs(), m_ConstantInt(&c));
      if (!constant) { input = mul.getRhs(); constant = matchPattern(mul.getLhs(), m_ConstantInt(&c)); }
      if (constant && c.isPowerOf2() && c.logBase2() <= shift) return recurse(input, shift - c.logBase2());
    }
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
      Value input = add.getLhs();
      bool constant = matchPattern(add.getRhs(), m_ConstantInt(&c));
      if (!constant) { input = add.getRhs(); constant = matchPattern(add.getLhs(), m_ConstantInt(&c)); }
      if (constant) {
        unsigned zeros = trailingZeros(input);
        if (zeros >= shift + width) return number(c.lshr(shift).getZExtValue() & mask);
        if (zeros >= shift && c.getActiveBits() <= shift) return recurse(input, shift);
      }
    }
  }
  Value result = value;
  if (shift) result = arith::ShRUIOp::create(b, loc, result, number(shift));
  return arith::AndIOp::create(b, loc, result, number(mask));
}

// Keep dynamically indexed SSA snapshots in addressable storage. Subbyte
// carriers need native bytes; wide vectors need one shared private copy,
// rather than LLVM independently materializing the full value at each extract.
class IndexedValues : public PassWrapper<IndexedValues, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexedValues)
  StringRef getArgument() const override { return "weft-x86-materialize-indexed-values"; }
  StringRef getDescription() const override {
    return "Materialize shared addressable snapshots for dynamic vector projections";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    SmallVector<LLVM::ExtractElementOp> extracts;
    getOperation().walk([&](LLVM::ExtractElementOp op) {
      auto type = op.getVector().getType(); APInt index;
      auto function = op->getParentOfType<func::FuncOp>();
      auto bits = function ? function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits") : IntegerAttr{};
      bool numeric = isa<IntegerType, FloatType>(type.getElementType());
      bool subbyte = numeric && type.getElementTypeBitWidth() < 8;
      bool wide = numeric && bits && type.getNumElements() * type.getElementTypeBitWidth() > bits.getInt();
      if (type.getRank() == 1 && !type.isScalable() && (subbyte || wide) &&
          !matchPattern(op.getPosition(), m_ConstantInt(&index)))
        extracts.push_back(op);
    });
    DenseMap<Value, Value> snapshots;
    DenseMap<Value, SmallVector<LLVM::ExtractElementOp>> demands;
    for (auto extract : extracts) demands[extract.getVector()].push_back(extract);
    auto conditionalCopyRegion = [&](Value source) -> Region * {
      Region *common = nullptr;
      for (auto extract : demands.lookup(source)) {
        Operation *edge = extract;
        Region *region = nullptr;
        while (edge && edge->getBlock() != source.getParentBlock()) {
          auto *parentRegion = edge->getParentRegion();
          auto *parent = edge->getParentOp();
          if (parent && parent->getBlock() == source.getParentBlock()) {
            if (isa<scf::IfOp>(parent)) region = parentRegion;
            break;
          }
          edge = parent;
        }
        if (!region || (common && common != region)) return nullptr;
        common = region;
      }
      return common;
    };
    DenseMap<Value, std::pair<int64_t, int64_t>> compressed;
    PredicateLanes predicates;
    for (auto extract : extracts) {
      Value source = extract.getVector(), storage = snapshots.lookup(source);
      auto type = extract.getVector().getType(); Location loc = extract.getLoc();
      if (!storage) {
        auto function = extract->getParentOfType<func::FuncOp>();
        SmallVector<int64_t> representatives;
        int64_t repeat = 1;
        if (type.getElementType().isInteger(1)) representatives = predicates.representatives(source, repeat);
        int64_t count = representatives.empty() ? type.getNumElements() : representatives.size();
        b.setInsertionPointToStart(&function.getBody().front());
        Type element = type.getElementTypeBitWidth() < 8 ? b.getI8Type() : type.getElementType();
        storage = memref::AllocaOp::create(b, loc, MemRefType::get({count}, element));
        if (Region *region = conditionalCopyRegion(source)) {
          // All indexed consumers are under this one conditional arm. Copy
          // the already saved SSA value once at its entry, outside its loops.
          // Do not delay the source read or move a copy into a repeated scope.
          b.setInsertionPointToStart(&region->front());
        } else if (auto argument = dyn_cast<BlockArgument>(source)) {
          if (argument.getOwner() == &function.getBody().front()) b.setInsertionPointAfter(storage.getDefiningOp());
          else b.setInsertionPointToStart(argument.getOwner());
        }
        else b.setInsertionPointAfter(source.getDefiningOp());
        Value bytes = source;
        if (!representatives.empty()) {
          bytes = vector::ShuffleOp::create(b, loc, source, source, representatives);
          compressed[source] = {repeat, count};
        }
        if (element != type.getElementType())
          bytes = arith::ExtUIOp::create(b, loc, cast<VectorType>(bytes.getType()).clone(element), bytes);
        Value zero = arith::ConstantIndexOp::create(b, loc, 0);
        vector::StoreOp::create(b, loc, bytes, storage, zero);
        snapshots[source] = storage;
      }
      b.setInsertionPoint(extract);
      Value index = arith::IndexCastOp::create(b, loc, b.getIndexType(), extract.getPosition());
      if (auto found = compressed.find(source); found != compressed.end()) {
        auto [repeat, period] = found->second;
        if (period == 1) index = arith::ConstantIndexOp::create(b, loc, 0);
        else if (llvm::isPowerOf2_64(repeat) && llvm::isPowerOf2_64(period))
          index = indexBits(index, llvm::Log2_64(repeat), llvm::Log2_64(period), b, loc);
        else {
          if (repeat != 1) index = arith::DivUIOp::create(b, loc, index, arith::ConstantIndexOp::create(b, loc, repeat));
          if (period * repeat < type.getNumElements())
            index = arith::RemUIOp::create(b, loc, index, arith::ConstantIndexOp::create(b, loc, period));
        }
      }
      Value byte = memref::LoadOp::create(b, loc, storage, index);
      Value value = byte;
      if (byte.getType() != type.getElementType()) value = arith::TruncIOp::create(b, loc, type.getElementType(), byte);
      b.replaceOp(extract, value);
    }
  }
};

struct IndexedLane {
  Value coordinates;
  int64_t lane, bias;
  bool unsignedSequence = true;
};

std::optional<int64_t> modularStep(Value value, unsigned depth = 0) {
  if (depth > 12) return {};
  DenseIntElementsAttr literal;
  if (matchPattern(value, m_Constant(&literal))) {
    auto numbers = literal.getValues<APInt>();
    if (literal.isSplat()) return 0;
    APInt previous = *numbers.begin();
    for (const APInt &number : llvm::drop_begin(numbers)) {
      if (!(number - previous).isOne()) return {};
      previous = number;
    }
    return 1;
  }
  if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>())
    if (!isa<VectorType>(broadcast.getSource().getType())) return 0;
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto a = modularStep(add.getLhs(), depth + 1), c = modularStep(add.getRhs(), depth + 1);
    if (a && c && *a + *c <= 1) return *a + *c;
  }
  return {};
}

// This relation only groups candidate reads. Actual extended coordinates are
// checked before a packet load, so narrow wrap and inactive safe indices retain
// their individual-read behavior.
std::optional<IndexedLane> indexedLane(Value index, unsigned depth = 0) {
  if (depth > 12) return {};
  if (auto extract = index.getDefiningOp<vector::ExtractOp>()) {
    if (extract.hasDynamicPosition() || extract.getStaticPosition().size() != 1) return {};
    return IndexedLane{extract.getSource(), extract.getStaticPosition()[0], 0};
  }
  Operation *op = index.getDefiningOp();
  if (isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtUIOp, arith::ExtSIOp>(op)) {
    auto result = indexedLane(op->getOperand(0), depth + 1);
    if (result && (indexWidth(index.getType()) < indexWidth(op->getOperand(0).getType()) || isa<arith::ExtSIOp>(op) ||
        (isa<arith::IndexCastOp>(op) && indexWidth(op->getOperand(0).getType()) < 64)))
      result->unsignedSequence = false;
    return result;
  }
  if (auto add = dyn_cast_or_null<arith::AddIOp>(op)) {
    APInt offset;
    Value source = add.getLhs();
    if (!matchPattern(add.getRhs(), m_ConstantInt(&offset))) {
      source = add.getRhs();
      if (!matchPattern(add.getLhs(), m_ConstantInt(&offset))) return {};
    }
    if (offset.isNegative() || offset.getActiveBits() > 62) return {};
    auto result = indexedLane(source, depth + 1);
    if (!result || result->bias > INT64_MAX - offset.getSExtValue()) return {};
    result->bias += offset.getSExtValue();
    if (indexWidth(index.getType()) < 64) result->unsignedSequence = false;
    return result;
  }
  if (auto select = dyn_cast_or_null<arith::SelectOp>(op)) {
    APInt inactive;
    if (matchPattern(select.getFalseValue(), m_ConstantInt(&inactive)) && inactive.isZero()) {
      auto result = indexedLane(select.getTrueValue(), depth + 1);
      if (result) result->unsignedSequence = false;
      return result;
    }
  }
  return {};
}

class IndexedPackets : public PassWrapper<IndexedPackets, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexedPackets)
  StringRef getArgument() const override { return "weft-x86-coalesce-indexed-packets"; }
  StringRef getDescription() const override { return "Coalesce contiguous reads from private indexed snapshots"; }
  void runOnOperation() override {
    SmallVector<memref::LoadOp> loads;
    getOperation().walk([&](memref::LoadOp load) {
      auto type = load.getMemRefType();
      if (type.getRank() == 1 && type.hasStaticShape() &&
          load.getMemref().getDefiningOp<memref::AllocaOp>() &&
          type.getElementType().isIntOrFloat() && indexedLane(load.getIndices()[0])) loads.push_back(load);
    });
    IRRewriter b(&getContext());
    DominanceInfo dominance(getOperation());
    llvm::SmallPtrSet<Operation *, 32> retired;
    int64_t selected = 0;
    for (size_t first = 0; first < loads.size(); ++first) {
      auto begin = loads[first];
      if (retired.contains(begin)) continue;
      auto relation = indexedLane(begin.getIndices()[0]);
      if (!relation) continue;
      auto function = begin->getParentOfType<func::FuncOp>();
      auto bits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits");
      if (!bits) continue;
      auto storageType = begin.getMemRefType();
      int64_t elementBits = storageType.getElementTypeBitWidth();
      int64_t capacity = bits.getInt() / elementBits;
      SmallVector<memref::LoadOp> group{begin};
      bool unitSequence = relation->unsignedSequence && modularStep(relation->coordinates) == std::optional<int64_t>(1);
      for (size_t next = first + 1; next < loads.size() && int64_t(group.size()) < capacity; ++next) {
        auto load = loads[next];
        if (retired.contains(load)) continue;
        if (load->getBlock() != begin->getBlock()) break;
        if (load.getMemref() != begin.getMemref()) continue;
        auto current = indexedLane(load.getIndices()[0]);
        if (!current || current->coordinates != relation->coordinates || current->bias != relation->bias ||
            current->lane != relation->lane + int64_t(group.size())) break;
        group.push_back(load);
        unitSequence &= current->unsignedSequence;
      }
      if (group.size() * elementBits < 128 || int64_t(group.size()) > storageType.getDimSize(0)) continue;
      bool safe = true;
      for (Operation *op = begin->getNextNode(); op != group.back(); op = op->getNextNode())
        if (op->getNumRegions() || (!isMemoryEffectFree(op) && !isa<memref::LoadOp, vector::LoadOp>(op))) {
          safe = false; break;
        }
      if (!safe) continue;
      llvm::SmallPtrSet<Operation *, 32> seen;
      SmallVector<Operation *> addresses;
      auto collect = [&](auto &&self, Value value) -> bool {
        if (dominance.dominates(value, begin)) return true;
        Operation *op = value.getDefiningOp();
        if (!op || op->getBlock() != begin->getBlock() || op->getNumRegions() ||
            !isMemoryEffectFree(op) || !isSpeculatable(op) || seen.size() >= 256) return false;
        if (!seen.insert(op).second) return true;
        for (Value operand : op->getOperands()) if (!self(self, operand)) return false;
        addresses.push_back(op);
        return true;
      };
      for (auto load : group) if (!collect(collect, load.getIndices()[0])) { safe = false; break; }
      if (!safe) continue;
      b.setInsertionPoint(begin);
      Location loc = begin.getLoc();
      IRMapping mapping;
      for (Operation *op : addresses) b.clone(*op, mapping);
      SmallVector<Value> indices;
      for (auto load : group) indices.push_back(mapping.lookupOrDefault(load.getIndices()[0]));
      int64_t width = group.size();
      Value limit = arith::ConstantIndexOp::create(b, loc, storageType.getDimSize(0) - width);
      Value contiguous = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, indices.front(), limit);
      auto coordinateType = cast<VectorType>(relation->coordinates.getType()).getElementType();
      unsigned coordinateBits = indexWidth(coordinateType);
      unitSequence &= coordinateBits >= 63 || uint64_t(width) <= (uint64_t(1) << coordinateBits);
      if (unitSequence) {
        // A modular unit sequence is contiguous after unsigned extension when
        // its first coordinate leaves room for the entire packet before wrap.
        Value head = vector::ExtractOp::create(b, loc, relation->coordinates, ArrayRef<int64_t>{relation->lane});
        APInt last = APInt::getMaxValue(coordinateBits) - (width - 1);
        Value bound = arith::ConstantOp::create(b, loc, b.getIntegerAttr(coordinateType, last));
        contiguous = arith::AndIOp::create(b, loc, contiguous,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, head, bound));
      } else {
        auto indexType = VectorType::get({width}, b.getIndexType());
        SmallVector<int64_t> lanes;
        for (int64_t lane = 0; lane < width; ++lane) lanes.push_back(lane);
        Value offsets = arith::ConstantOp::create(b, loc, indexType, DenseIntElementsAttr::get(indexType, lanes));
        Value expected = arith::AddIOp::create(b, loc, offsets, vector::BroadcastOp::create(b, loc, indexType, indices.front()));
        Value actual = vector::FromElementsOp::create(b, loc, indexType, indices);
        Value matches = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, actual, expected);
        contiguous = arith::AndIOp::create(b, loc, contiguous,
            vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, matches));
      }
      auto packetType = VectorType::get({width}, storageType.getElementType());
      auto choice = scf::IfOp::create(b, loc, TypeRange{packetType}, contiguous, true);
      b.setInsertionPointToStart(choice.thenBlock());
      Value packet = vector::LoadOp::create(b, loc, packetType, begin.getMemref(), indices.front());
      scf::YieldOp::create(b, loc, packet);
      b.setInsertionPointToStart(choice.elseBlock());
      SmallVector<Value> scalar;
      for (Value index : indices) scalar.push_back(memref::LoadOp::create(b, loc, begin.getMemref(), index));
      scf::YieldOp::create(b, loc, vector::FromElementsOp::create(b, loc, packetType, scalar).getResult());
      for (auto [lane, load] : llvm::enumerate(group)) {
        b.setInsertionPoint(load);
        Value element = vector::ExtractOp::create(b, loc, choice.getResult(0), ArrayRef<int64_t>{int64_t(lane)});
        retired.insert(load);
        b.replaceOp(load, element);
      }
      ++selected;
    }
    if (selected) getOperation()->setAttr("weft.x86.indexed_packets", b.getI64IntegerAttr(selected));
  }
};
} // namespace
bool weft::x86::hasUniformPredicateGroups(Value predicate, int64_t group) {
  auto type = dyn_cast<VectorType>(predicate.getType());
  if (!type || type.isScalable() || type.getRank() != 1 || !type.getElementType().isInteger(1) ||
      group < 1 || type.getNumElements() % group) return false;
  PredicateLanes lanes;
  for (int64_t start = 0; start < type.getNumElements(); start += group) {
    unsigned first = lanes.identify(predicate, start);
    for (int64_t lane = 1; lane < group; ++lane)
      if (lanes.identify(predicate, start + lane) != first) return false;
  }
  return true;
}
std::unique_ptr<Pass> weft::createX86IndexedValuePass() { return std::make_unique<IndexedValues>(); }
std::unique_ptr<Pass> weft::createX86IndexedPacketPass() { return std::make_unique<IndexedPackets>(); }
