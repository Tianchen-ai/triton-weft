#include "Lookups.h"
#include "Weft/Target/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {

Value intrinsic(ModuleOp module, OpBuilder &b, Location loc, StringRef name,
                VectorType type, Value table, Value indices) {
  auto declaration = module.lookupSymbol<func::FuncOp>(name);
  if (!declaration) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    declaration = func::FuncOp::create(b, loc, name,
        b.getFunctionType(TypeRange{type, type}, TypeRange{type}));
    declaration.setPrivate();
  }
  return func::CallOp::create(b, loc, declaration, ValueRange{table, indices}).getResult(0);
}

Value integerWidth(OpBuilder &b, Location loc, Value value, unsigned bits) {
  auto type = cast<VectorType>(value.getType());
  auto result = type.clone(b.getIntegerType(bits));
  unsigned original = type.getElementTypeBitWidth();
  if (original == bits) return value;
  if (original < bits) return arith::ExtUIOp::create(b, loc, result, value);
  return arith::TruncIOp::create(b, loc, result, value);
}

Value slice(OpBuilder &b, Location loc, Value value, int64_t start, int64_t size) {
  SmallVector<int64_t> lanes;
  for (int64_t i = 0; i < size; ++i) lanes.push_back(start + i);
  return vector::ShuffleOp::create(b, loc, value, value, lanes);
}

Value pad(OpBuilder &b, Location loc, Value value, int64_t count) {
  auto original = cast<VectorType>(value.getType());
  if (original.getNumElements() == count) return value;
  auto type = VectorType::get({count}, original.getElementType());
  Value zeros = arith::ConstantOp::create(b, loc, type,
      DenseElementsAttr::get(type, b.getIntegerAttr(type.getElementType(), 0)));
  return vector::InsertStridedSliceOp::create(b, loc, value, zeros,
      ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
}

struct ByteLane {
  Value value;
  int64_t index;
};

std::optional<ByteLane> byteLane(Value value, int64_t index,
                                llvm::SetVector<Operation *> *assembly = nullptr) {
  for (unsigned depth = 0; depth != 64; ++depth) {
    auto type = dyn_cast<VectorType>(value.getType());
    if (!type || type.getRank() != 1 || type.isScalable() ||
        !type.getElementType().isInteger(8) || index < 0 || index >= type.getNumElements()) return {};
    if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
      if (assembly) assembly->insert(shuffle);
      int64_t count = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
      index = shuffle.getMask()[index];
      if (index < 0) return {};
      value = index < count ? shuffle.getV1() : shuffle.getV2();
      if (index >= count) index -= count;
    } else if (auto insert = value.getDefiningOp<vector::InsertStridedSliceOp>()) {
      if (insert.getOffsets().size() != 1 || cast<IntegerAttr>(insert.getStrides()[0]).getInt() != 1) return {};
      if (assembly) assembly->insert(insert);
      int64_t offset = cast<IntegerAttr>(insert.getOffsets()[0]).getInt();
      int64_t count = cast<VectorType>(insert.getValueToStore().getType()).getNumElements();
      if (index >= offset && index < offset + count) {
        value = insert.getValueToStore(); index -= offset;
      } else value = insert.getDest();
    } else if (auto extract = value.getDefiningOp<vector::ExtractStridedSliceOp>()) {
      if (extract.getOffsets().size() != 1 || cast<IntegerAttr>(extract.getStrides()[0]).getInt() != 1) return {};
      if (assembly) assembly->insert(extract);
      index += cast<IntegerAttr>(extract.getOffsets()[0]).getInt();
      value = extract.getSource();
    } else return ByteLane{value, index};
  }
  return {};
}

bool repeatedByteTable(Value value) {
  auto type = cast<VectorType>(value.getType());
  for (int64_t i = 16; i < type.getNumElements(); ++i) {
    auto lane = byteLane(value, i), base = byteLane(value, i % 16);
    if (!lane || !base) return false;
    if (lane->value == base->value && lane->index == base->index) continue;
    auto a = lane->value.getDefiningOp<arith::ConstantOp>();
    auto c = base->value.getDefiningOp<arith::ConstantOp>();
    if (!a || !c) return false;
    auto av = dyn_cast<DenseIntElementsAttr>(a.getValue());
    auto cv = dyn_cast<DenseIntElementsAttr>(c.getValue());
    if (!av || !cv || av.getValues<APInt>()[lane->index] != cv.getValues<APInt>()[base->index]) return false;
  }
  return true;
}

// A table lookup commutes with a permutation of its result when all native
// 128-bit table lanes hold the same snapshot. Rebuild the complete consumer
// cohort, so this never adds lookups while retaining another live full result.
class ProjectByteLookups : public RewritePattern {
public:
  explicit ProjectByteLookups(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *root, PatternRewriter &b) const override {
    if (!isa<vector::InsertStridedSliceOp, vector::ShuffleOp>(root) || root->use_empty()) return failure();
    auto type = dyn_cast<VectorType>(root->getResult(0).getType());
    if (!type || type.getRank() != 1 || type.isScalable() || !type.getElementType().isInteger(8) ||
        type.getNumElements() <= 32 || type.getNumElements() > 4096) return failure();
    llvm::SetVector<Operation *> consumers, assembly, calls;
    for (Operation *user : root->getUsers()) {
      if (!isa<vector::ShuffleOp>(user) || user->getBlock() != root->getBlock()) return failure();
      consumers.insert(user);
    }
    Value table;
    StringRef callee;
    int64_t packet = 0, issues = 0;
    bool crossesPackets = false;
    SmallVector<SmallVector<ByteLane>> selections;
    for (Operation *consumer : consumers) {
      auto shuffle = cast<vector::ShuffleOp>(consumer);
      int64_t first = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
      SmallVector<ByteLane> selected;
      Operation *packetSource = nullptr;
      for (int64_t position : shuffle.getMask()) {
        if (position < 0 || (position < first ? shuffle.getV1() : shuffle.getV2()) != root->getResult(0))
          return failure();
        auto output = byteLane(root->getResult(0), position < first ? position : position - first, &assembly);
        if (!output) return failure();
        auto call = output->value.getDefiningOp<func::CallOp>();
        if (!call || call->getBlock() != root->getBlock() || call.getNumOperands() != 2 ||
            (call.getCallee() != "llvm.x86.avx2.pshuf.b" &&
             call.getCallee() != "llvm.x86.avx512.pshuf.b.512")) return failure();
        if (table && (table != call.getOperand(0) || callee != call.getCallee())) return failure();
        table = call.getOperand(0); callee = call.getCallee();
        packet = cast<VectorType>(table.getType()).getNumElements();
        if (selected.size() % packet == 0) packetSource = call;
        else crossesPackets |= packetSource != call.getOperation();
        auto input = byteLane(call.getOperand(1), output->index);
        if (!input) return failure();
        selected.push_back(*input);
        calls.insert(call); assembly.insert(call);
      }
      issues += llvm::divideCeil<int64_t>(selected.size(), packet);
      selections.push_back(std::move(selected));
    }
    if (!table || !crossesPackets || issues > static_cast<int64_t>(calls.size()) ||
        !repeatedByteTable(table)) return failure();
    for (const auto &selected : selections)
      for (const ByteLane &lane : selected)
        if (assembly.contains(lane.value.getDefiningOp())) return failure();
    for (Operation *op : assembly)
      for (Operation *user : op->getUsers())
        if (!assembly.contains(user) && !consumers.contains(user)) return failure();
    auto packetType = cast<VectorType>(table.getType());
    auto module = root->getParentOfType<ModuleOp>();
    for (auto [i, consumer] : llvm::enumerate(consumers)) {
      b.setInsertionPoint(consumer);
      Location loc = consumer->getLoc();
      auto outputType = cast<VectorType>(consumer->getResult(0).getType());
      Value result = arith::ConstantOp::create(b, loc, outputType,
          DenseElementsAttr::get(outputType, b.getI8IntegerAttr(0)));
      for (int64_t start = 0; start < outputType.getNumElements(); start += packet) {
        int64_t count = std::min(packet, outputType.getNumElements() - start);
        Value chosen = arith::ConstantOp::create(b, loc, packetType,
            DenseElementsAttr::get(packetType, b.getI8IntegerAttr(0)));
        // Preserve already packetized integer producers. Each contiguous run
        // references its original SSA indices; later integer projection can
        // shrink their disjoint demands without reconstructing a wide value.
        for (int64_t begin = 0; begin < count;) {
          Value source = selections[i][start + begin].value;
          SmallVector<int64_t> mask;
          int64_t end = begin;
          while (end < count && selections[i][start + end].value == source)
            mask.push_back(selections[i][start + end++].index);
          Value part = vector::ShuffleOp::create(b, loc, source, source, mask);
          chosen = vector::InsertStridedSliceOp::create(b, loc, part, chosen,
              ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
          begin = end;
        }
        Value decoded = intrinsic(module, b, loc, callee, packetType, table, chosen);
        if (count != packet) decoded = slice(b, loc, decoded, 0, count);
        result = vector::InsertStridedSliceOp::create(b, loc, decoded, result,
            ArrayRef<int64_t>{start}, ArrayRef<int64_t>{1});
      }
      b.replaceOp(consumer, result);
    }
    // Only the proven pure permutations and PSHUFB calls are removed. The table
    // read and index producers retain their original snapshots and effects.
    while (!assembly.empty()) {
      Operation *dead = nullptr;
      for (Operation *op : assembly) if (op->use_empty()) { dead = op; break; }
      assert(dead && "closed lookup assembly must become dead after replacing all consumers");
      assembly.remove(dead); b.eraseOp(dead);
    }
    return success();
  }
};

class ProjectLookupLayouts : public PassWrapper<ProjectLookupLayouts, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProjectLookupLayouts)
  StringRef getArgument() const override { return "weft-x86-project-lookup-layouts"; }
  StringRef getDescription() const override { return "Propagate consumer byte layouts through replicated register table lookups"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ProjectByteLookups>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> weft::createX86LookupProjectionPass() {
  return std::make_unique<ProjectLookupLayouts>();
}

Value weft::x86::registerLookup(ModuleOp module, OpBuilder &b, Location loc,
                                Value table, Value indices, unsigned vectorBits) {
  auto tableType = dyn_cast<VectorType>(table.getType());
  if (!tableType || tableType.isScalable() || tableType.getRank() != 1) return {};
  int64_t entries = tableType.getNumElements();
  Type element = tableType.getElementType();
  unsigned bits = element.getIntOrFloatBitWidth();
  bool bytes = isa<IntegerType>(element) && bits <= 8 && entries <= 16;
  int64_t dwords = vectorBits / 32;
  if (!bytes && (bits > 32 || entries > dwords)) return {};

  // PSHUFB has independent 128-bit lanes: repeat the same <=16-entry table
  // in each lane. PERMD instead addresses the entire 256/512-bit register.
  // Neither operation changes the bits of the selected finite-width value.
  unsigned carrierBits = bytes ? 8 : 32;
  auto indexType = dyn_cast<VectorType>(indices.getType());
  bool scalarResult = !indexType;
  if (scalarResult)
    indices = vector::FromElementsOp::create(b, loc,
        VectorType::get({1}, indices.getType()), ValueRange{indices});
  int64_t count = cast<VectorType>(indices.getType()).getNumElements();
  // A single narrower issue already covers this consumer and the complete
  // table. Do not construct a 512-bit repeated table only to discard the upper
  // half of the lookup result; native lowering can retain that unnecessary
  // preparation even after narrowing the shuffle itself.
  if (vectorBits == 512 && count <= 256 / carrierBits &&
      (bytes || entries <= 8))
    vectorBits = 256;
  int64_t packet = vectorBits / carrierBits;

  if (element.isInteger(1)) {
    // Keep predicate lookup as integer arithmetic. An opaque byte-shuffle
    // intrinsic hides a uniform true table from constant propagation and
    // leaves masks on otherwise complete dot-product packets. The source
    // in-bounds contract limits indices to the original table; inactive
    // indices have already been replaced by zero by the caller.
    auto maskType = VectorType::get({32}, b.getI1Type());
    Value mask = arith::ConstantOp::create(b, loc, maskType,
        DenseElementsAttr::get(maskType, b.getBoolAttr(true)));
    mask = vector::InsertStridedSliceOp::create(b, loc, table, mask,
        ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
    Value word = vector::BitCastOp::create(b, loc, VectorType::get({1}, b.getI32Type()), mask);
    word = vector::ExtractOp::create(b, loc, word, ArrayRef<int64_t>{0});
    auto words = VectorType::get({count}, b.getI32Type());
    Value value = vector::BroadcastOp::create(b, loc, words, word);
    Value positions = integerWidth(b, loc, indices, 32);
    Value limit = arith::ConstantOp::create(b, loc, words,
        DenseElementsAttr::get(words, b.getI32IntegerAttr(31)));
    positions = arith::AndIOp::create(b, loc, positions, limit);
    value = arith::ShRUIOp::create(b, loc, value, positions);
    value = arith::TruncIOp::create(b, loc, VectorType::get({count}, b.getI1Type()), value);
    if (scalarResult) value = vector::ExtractOp::create(b, loc, value, ArrayRef<int64_t>{0});
    return value;
  }

  Value payload = table;
  if (isa<FloatType>(element))
    payload = arith::BitcastOp::create(b, loc, tableType.clone(b.getIntegerType(bits)), payload);
  payload = integerWidth(b, loc, payload, carrierBits);
  if (bytes) {
    payload = pad(b, loc, payload, 16);
    SmallVector<int64_t> lanes;
    for (int64_t i = 0; i < packet; ++i) lanes.push_back(i % 16);
    payload = vector::ShuffleOp::create(b, loc, payload, payload, lanes);
  } else payload = pad(b, loc, payload, packet);

  indices = integerWidth(b, loc, indices, carrierBits);
  StringRef name = bytes ? (vectorBits == 512 ? "llvm.x86.avx512.pshuf.b.512" : "llvm.x86.avx2.pshuf.b")
                         : (vectorBits == 512 ? "llvm.x86.avx512.permvar.si.512" : "llvm.x86.avx2.permd");
  auto packetType = cast<VectorType>(payload.getType());
  auto outputType = VectorType::get({count}, b.getIntegerType(carrierBits));
  Value result = arith::ConstantOp::create(b, loc, outputType,
      DenseElementsAttr::get(outputType, b.getIntegerAttr(outputType.getElementType(), 0)));
  for (int64_t start = 0; start < count; start += packet) {
    int64_t length = std::min(packet, count - start);
    Value selectedIndices = pad(b, loc, slice(b, loc, indices, start, length), packet);
    Value selected = intrinsic(module, b, loc, name, packetType, payload, selectedIndices);
    if (length != packet) selected = slice(b, loc, selected, 0, length);
    result = vector::InsertStridedSliceOp::create(b, loc, selected, result,
        ArrayRef<int64_t>{start}, ArrayRef<int64_t>{1});
  }
  result = integerWidth(b, loc, result, bits);
  if (isa<FloatType>(element))
    result = arith::BitcastOp::create(b, loc, VectorType::get({count}, element), result);
  if (scalarResult) result = vector::ExtractOp::create(b, loc, result, ArrayRef<int64_t>{0});
  return result;
}
