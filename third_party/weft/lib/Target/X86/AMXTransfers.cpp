#include "AMXContractions.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
namespace {
Value allocation(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>()) value = cast.getSource();
  return value.getDefiningOp<memref::AllocOp>() || value.getDefiningOp<memref::AllocaOp>() ? value : Value();
}

bool allTrue(Value value) {
  DenseElementsAttr elements;
  return matchPattern(value, m_Constant(&elements)) && elements.isSplat() &&
         elements.getElementType().isInteger(1) && elements.getSplatValue<APInt>().isOne();
}
bool conditionProvesMask(Value condition, Value mask, unsigned depth = 0) {
  if (depth > 16) return false;
  if (auto reduction = condition.getDefiningOp<vector::ReductionOp>())
    return reduction.getKind() == vector::CombiningKind::AND && reduction.getVector() == mask;
  if (auto both = condition.getDefiningOp<arith::AndIOp>())
    return conditionProvesMask(both.getLhs(), mask, depth + 1) ||
           conditionProvesMask(both.getRhs(), mask, depth + 1);
  return false;
}
bool activeAt(Value mask, Operation *at) {
  if (allTrue(mask)) return true;
  for (Operation *parent = at->getParentOp(); parent; parent = parent->getParentOp())
    if (auto choice = dyn_cast<scf::IfOp>(parent))
      if (choice.getThenRegion().isAncestor(at->getParentRegion()) && conditionProvesMask(choice.getCondition(), mask))
        return true;
  return false;
}

struct Lane {
  LLVM::MaskedLoadOp load;
  int64_t index;
};

// Follow movement of an existing byte snapshot, including casts that preserve
// its low byte. Never substitute arithmetic or a fresh read for the snapshot.
std::optional<Lane> lane(Value value, int64_t index, Operation *at, unsigned depth = 0) {
  auto type = dyn_cast<VectorType>(value.getType());
  if (depth > 96 || !type || type.getRank() != 1 || !type.getElementType().isInteger() ||
      type.getElementTypeBitWidth() < 8 ||
      index < 0 || index >= type.getNumElements()) return {};
  if (auto load = value.getDefiningOp<LLVM::MaskedLoadOp>()) {
    if (!type.getElementType().isInteger(8) || !activeAt(load.getMask(), at)) return {};
    return Lane{load, index};
  }
  if (isa_and_nonnull<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(value.getDefiningOp()))
    return lane(value.getDefiningOp()->getOperand(0), index, at, depth + 1);
  if (auto mask = value.getDefiningOp<arith::AndIOp>()) {
    for (unsigned side = 0; side != 2; ++side) {
      DenseElementsAttr constant;
      if (matchPattern(mask.getOperand(side), m_Constant(&constant)) && constant.isSplat() &&
          constant.getSplatValue<APInt>().extractBits(8, 0).isAllOnes())
        return lane(mask.getOperand(1 - side), index, at, depth + 1);
    }
  }
  if (auto insert = value.getDefiningOp<vector::InsertStridedSliceOp>()) {
    auto offsets = insert.getOffsets(), strides = insert.getStrides();
    if (offsets.size() != 1 || strides.size() != 1 || cast<IntegerAttr>(strides[0]).getInt() != 1) return {};
    int64_t first = cast<IntegerAttr>(offsets[0]).getInt();
    int64_t count = cast<VectorType>(insert.getOperand(0).getType()).getNumElements();
    return index >= first && index < first + count
        ? lane(insert.getOperand(0), index - first, at, depth + 1)
        : lane(insert.getDest(), index, at, depth + 1);
  }
  if (auto extract = value.getDefiningOp<vector::ExtractStridedSliceOp>()) {
    if (extract.getOffsets().size() != 1 || extract.getStrides().size() != 1 ||
        cast<IntegerAttr>(extract.getStrides()[0]).getInt() != 1) return {};
    return lane(extract.getSource(), index + cast<IntegerAttr>(extract.getOffsets()[0]).getInt(), at, depth + 1);
  }
  if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
    int64_t selected = shuffle.getMask()[index];
    int64_t count = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
    return selected < count ? lane(shuffle.getV1(), selected, at, depth + 1)
                            : lane(shuffle.getV2(), selected - count, at, depth + 1);
  }
  return {};
}

// Modular pointer-width linear forms prove row spacing even when each row's
// address was independently assembled by encoded storage lowering.
struct Linear {
  uint64_t constant = 0;
  llvm::DenseMap<Value, uint64_t> terms;
  void add(const Linear &other, uint64_t scale = 1) {
    constant += other.constant * scale;
    for (auto [value, coefficient] : other.terms) {
      auto &result = terms[value];
      result += coefficient * scale;
      if (!result) terms.erase(value);
    }
  }
  bool operator==(const Linear &other) const {
    if (constant != other.constant || terms.size() != other.terms.size()) return false;
    for (auto [value, coefficient] : terms)
      if (other.terms.lookup(value) != coefficient) return false;
    return true;
  }
};
Linear linear(Value value, unsigned depth = 0) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) { Linear result; result.constant = constant.getZExtValue(); return result; }
  if (depth < 32) {
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      if (cast.getIn().getType().isIndex() && cast.getOut().getType().isInteger(64))
        return linear(cast.getIn(), depth + 1);
    }
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
      Linear result = linear(add.getLhs(), depth + 1); result.add(linear(add.getRhs(), depth + 1)); return result;
    }
    if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
      Linear result = linear(sub.getLhs(), depth + 1); result.add(linear(sub.getRhs(), depth + 1), uint64_t(-1)); return result;
    }
    if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
      Value input;
      if (matchPattern(mul.getRhs(), m_ConstantInt(&constant))) input = mul.getLhs();
      else if (matchPattern(mul.getLhs(), m_ConstantInt(&constant))) input = mul.getRhs();
      if (input) { Linear result; result.add(linear(input, depth + 1), constant.getZExtValue()); return result; }
    }
  }
  Linear result; result.terms[value] = 1; return result;
}

// The proof permits writes only to fresh, nonescaping private allocations.
bool privateBuffer(Value value) {
  value = allocation(value);
  if (!value) return false;
  SmallVector<Value> pending{value};
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (auto cast = dyn_cast<memref::CastOp>(user)) { pending.push_back(cast.getResult()); continue; }
      if (isa<vector::LoadOp, vector::StoreOp, memref::LoadOp, memref::StoreOp, memref::DimOp>(user)) continue;
      if (auto call = dyn_cast<func::CallOp>(user)) {
        auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
        if (function && function->hasAttr("weft.x86.amx_transfer")) continue;
      }
      return false;
    }
  }
  return true;
}
bool preservesExternalMemory(Operation *op) {
  if (isMemoryEffectFree(op) || isa<memref::AllocOp, memref::AllocaOp, LLVM::MaskedLoadOp>(op)) return true;
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return loop->hasAttr("weft.x86.amx_contraction") &&
        llvm::all_of(*loop.getBody(), [](Operation &body) { return preservesExternalMemory(&body); });
  if (auto store = dyn_cast<vector::StoreOp>(op)) return privateBuffer(store.getBase());
  if (auto store = dyn_cast<memref::StoreOp>(op)) return privateBuffer(store.getMemRef());
  if (auto call = dyn_cast<func::CallOp>(op)) {
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
    if (function && function->hasAttr("weft.x86.amx_transfer")) return privateBuffer(call.getOperand(0));
    StringRef name = call.getCallee();
    return name == "llvm.x86.tilezero" || name == "llvm.x86.ldtilecfg" ||
        name == "llvm.x86.tilerelease" || name == "llvm.x86.tileloadd64" || name == "llvm.x86.tdpbssd" ||
        name == "llvm.x86.tdpbsud" || name == "llvm.x86.tdpbusd" ||
        name == "llvm.x86.tdpbuud" || name == "llvm.x86.tdpbf16ps";
  }
  return false;
}
bool canDelay(Operation *load, Operation *use) {
  Operation *cursor = use;
  while (cursor->getBlock() != load->getBlock()) {
    for (Operation &op : *cursor->getBlock()) {
      if (&op == cursor) break;
      if (!preservesExternalMemory(&op)) return false;
    }
    cursor = cursor->getParentOp();
    if (!cursor) return false;
    if (auto loop = dyn_cast<scf::ForOp>(cursor)) {
      if (!loop->hasAttr("weft.x86.amx_contraction") ||
          !llvm::all_of(*loop.getBody(), [](Operation &op) { return preservesExternalMemory(&op); })) return false;
    } else if (!isa<scf::IfOp>(cursor)) return false;
  }
  if (!load->isBeforeInBlock(cursor)) return false;
  for (Operation *op = load->getNextNode(); op != cursor; op = op->getNextNode())
    if (!preservesExternalMemory(op)) return false;
  return true;
}

// Packing a saved SSA snapshot is independent of a traversal that reuses that
// snapshot. Move the private materialization as a unit, including its exact
// integer/permutation producers, without moving or repeating the source read.
class ReuseAMXPacking : public PassWrapper<ReuseAMXPacking, OperationPass<ModuleOp>> {
  int64_t deduplicate() {
    struct Packing {
      memref::AllocOp allocation;
      SmallVector<vector::StoreOp> stores;
    };
    SmallVector<memref::AllocOp> roots;
    llvm::SmallPtrSet<Operation *, 16> seen;
    getOperation().walk([&](vector::StoreOp store) {
      if (!store->hasAttr("weft.x86.amx_pack")) return;
      Value root = allocation(store.getBase());
      auto alloc = root ? root.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
      if (alloc && alloc->hasAttr("weft.x86.amx_pack_closed") && seen.insert(alloc).second) roots.push_back(alloc);
    });
    SmallVector<Packing> retained;
    int64_t merged = 0;
    for (auto alloc : roots) {
      Packing current{alloc, {}};
      SmallVector<Value> pending{alloc.getResult()};
      bool safe = true;
      while (!pending.empty() && safe)
        for (Operation *user : pending.pop_back_val().getUsers()) {
          if (auto cast = dyn_cast<memref::CastOp>(user)) pending.push_back(cast.getResult());
          else if (auto store = dyn_cast<vector::StoreOp>(user)) {
            if (!store->hasAttr("weft.x86.amx_pack") || store->getBlock() != alloc->getBlock()) { safe = false; break; }
            current.stores.push_back(store);
          } else if (!isa<func::CallOp>(user) || !user->hasAttr("weft.x86.amx_load")) { safe = false; break; }
        }
      if (!safe || current.stores.empty()) continue;
      llvm::sort(current.stores, [](auto lhs, auto rhs) { return lhs->isBeforeInBlock(rhs); });
      auto match = llvm::find_if(retained, [&](Packing &prior) {
        if (prior.allocation.getType() != alloc.getType() || prior.allocation->getBlock() != alloc->getBlock() ||
            !prior.stores.back()->isBeforeInBlock(alloc) || prior.stores.size() != current.stores.size() ||
            prior.allocation.getDynamicSizes() != alloc.getDynamicSizes()) return false;
        for (auto [a, c] : llvm::zip(prior.stores, current.stores))
          if (a.getValueToStore() != c.getValueToStore() || a.getIndices() != c.getIndices()) return false;
        return true;
      });
      if (match == retained.end()) { retained.push_back(std::move(current)); continue; }
      for (auto store : current.stores) store.erase();
      alloc.getResult().replaceAllUsesWith(match->allocation.getResult());
      alloc.erase(); ++merged;
    }
    return merged;
  }
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseAMXPacking)
  StringRef getArgument() const override { return "weft-x86-reuse-amx-packing"; }
  StringRef getDescription() const override { return "Reuse private tile packing within the original saved snapshot lifetime"; }
  void runOnOperation() override {
    SmallVector<memref::AllocOp> allocations;
    llvm::SmallPtrSet<Operation *, 16> seenAllocations;
    getOperation().walk([&](vector::StoreOp store) {
      if (!store->hasAttr("weft.x86.amx_pack")) return;
      Value root = allocation(store.getBase());
      auto alloc = root ? root.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
      if (alloc && seenAllocations.insert(alloc).second) allocations.push_back(alloc);
    });
    int64_t reused = 0;
    for (auto alloc : allocations) {
      alloc->removeAttr("weft.x86.amx_pack_closed");
      SmallVector<Value> pending{alloc.getResult()};
      SmallVector<Operation *> aliases;
      SmallVector<vector::StoreOp> stores;
      bool safe = true;
      while (!pending.empty() && safe)
        for (Operation *user : pending.pop_back_val().getUsers()) {
          if (auto cast = dyn_cast<memref::CastOp>(user)) { aliases.push_back(cast); pending.push_back(cast.getResult()); }
          else if (auto store = dyn_cast<vector::StoreOp>(user)) stores.push_back(store);
          else if (!isa<func::CallOp>(user) || !user->hasAttr("weft.x86.amx_load")) { safe = false; break; }
        }
      if (!safe || stores.empty()) continue;
      // The target materializer emits a complete set of constant-index stores.
      // Require every write to stay in the same birth block, with no other
      // writer or escaping use, before extending its physical residency.
      if (!llvm::all_of(stores, [&](auto store) { return store->getBlock() == alloc->getBlock(); })) continue;
      auto type = alloc.getType();
      int64_t count = type.hasStaticShape() ? type.getNumElements() : -1;
      APInt dynamic;
      if (count < 0 && alloc.getDynamicSizes().size() == 1 &&
          matchPattern(alloc.getDynamicSizes().front(), m_ConstantInt(&dynamic))) count = dynamic.getSExtValue();
      if (count <= 0 || count > 65536 / (type.getElementTypeBitWidth() / 8)) continue;
      llvm::SmallBitVector written(count, false);
      for (auto store : stores) {
        APInt offset;
        auto vector = store.getVectorType();
        if (store.getIndices().size() != 1 || !matchPattern(store.getIndices()[0], m_ConstantInt(&offset)) ||
            offset.isNegative() || offset.getZExtValue() + vector.getNumElements() > uint64_t(count)) { safe = false; break; }
        for (int64_t i = 0; i < vector.getNumElements(); ++i) {
          int64_t index = offset.getZExtValue() + i;
          if (written.test(index)) { safe = false; break; }
          written.set(index);
        }
      }
      if (!safe || !written.all()) continue;
      alloc->setAttr("weft.x86.amx_pack_closed", UnitAttr::get(&getContext()));
      scf::ForOp destination;
      SmallVector<Operation *> producers;
      for (auto loop = alloc->getParentOfType<scf::ForOp>(); loop; loop = loop->getParentOfType<scf::ForOp>()) {
        SmallVector<Value> roots(alloc.getDynamicSizes());
        for (auto store : stores) {
          roots.push_back(store.getValueToStore());
          llvm::append_range(roots, store.getIndices());
        }
        auto ordered = weft::analysis::movablePureSlice(roots, loop, [](Operation *op) {
          return llvm::all_of(op->getResultTypes(), [](Type type) { return getElementTypeOrSelf(type).isIntOrIndex(); });
        }, 4096);
        if (failed(ordered)) break;
        destination = loop;
        producers = std::move(*ordered);
      }
      if (!destination) continue;
      for (Operation *producer : producers) producer->moveBefore(destination);
      alloc->moveBefore(destination);
      for (Operation *alias : aliases) alias->moveBefore(destination);
      for (auto store : stores) store->moveBefore(destination);
      ++reused;
    }
    if (reused) getOperation()->setAttr("weft.x86.reused_amx_packings", IntegerAttr::get(IntegerType::get(&getContext(), 64), reused));
    if (int64_t merged = deduplicate())
      getOperation()->setAttr("weft.x86.shared_amx_packings", IntegerAttr::get(IntegerType::get(&getContext(), 64), merged));
  }
};

// Recognize a complete rectangular byte snapshot copied in native row packets.
// Only the address calculation is recreated at the tile read; the delay proof
// rejects intervening writes and opaque effects on the original input.
bool forwardPacketSnapshot(func::CallOp call, vector::StoreOp store,
                           int64_t first, Value reductionOffset, int64_t lastReduction,
                           int64_t stride, DominanceInfo &dominance) {
  auto columns = dyn_cast<scf::ForOp>(store->getParentOp());
  auto rows = columns ? dyn_cast<scf::ForOp>(columns->getParentOp()) : scf::ForOp();
  auto load = store.getValueToStore().getDefiningOp<LLVM::MaskedLoadOp>();
  auto type = cast<VectorType>(store.getValueToStore().getType());
  if (!columns || !rows || !load || load->getBlock() != store->getBlock() ||
      !columns->hasAttr("weft.x86.memory_packet_lanes") || !allTrue(load.getMask()) ||
      type.getRank() != 1 || !type.getElementType().isInteger(8) ||
      !columns.getInitArgs().empty() || !rows.getInitArgs().empty() ||
      store.getIndices().size() != 1 || !dominance.dominates(rows, call) ||
      !canDelay(rows, call)) return false;
  APInt low, high, step, rowHigh;
  if (!matchPattern(columns.getLowerBound(), m_ConstantInt(&low)) || !low.isZero() ||
      !matchPattern(columns.getUpperBound(), m_ConstantInt(&high)) ||
      !matchPattern(columns.getStep(), m_ConstantInt(&step)) ||
      high.getSExtValue() != stride || stride <= 0 ||
      step.getSExtValue() != type.getNumElements() || stride % type.getNumElements() ||
      !matchPattern(rows.getLowerBound(), m_Zero()) || !matchPattern(rows.getStep(), m_One()) ||
      !matchPattern(rows.getUpperBound(), m_ConstantInt(&rowHigh)) || rowHigh.getSExtValue() <= 0)
    return false;
  Linear expected;
  expected.terms[rows.getInductionVar()] = stride;
  expected.terms[columns.getInductionVar()] = 1;
  if (!(linear(store.getIndices()[0]) == expected)) return false;
  bool readOnly = true;
  rows.walk([&](Operation *op) {
    if (op != rows && op != columns && !preservesExternalMemory(op)) readOnly = false;
  });
  if (!readOnly) return false;
  auto geometry = call->getAttrOfType<DenseI64ArrayAttr>("weft.x86.amx_load");
  if (first < 0 || first / stride > rowHigh.getSExtValue() - geometry[0] ||
      first % stride + lastReduction + geometry[1] > stride) return false;
  auto gep = load.getData().getDefiningOp<LLVM::GEPOp>();
  if (!gep || !gep.getElemType().isInteger(8) || gep.getRawConstantIndices().size() != 1 ||
      gep.getDynamicIndices().size() != 1) return false;
  auto base = dyn_cast<BlockArgument>(gep.getBase());
  if (!base || !isa<func::FuncOp>(base.getOwner()->getParentOp())) return false;
  Value address = gep.getDynamicIndices().front();
  if (!address.getType().isInteger(64)) return false;
  SmallVector<Operation *> addressOps;
  llvm::SmallPtrSet<Operation *, 32> seen;
  std::function<bool(Value)> collect = [&](Value value) {
    if (value == rows.getInductionVar() || value == columns.getInductionVar() ||
        dominance.dominates(value, call)) return true;
    auto *op = value.getDefiningOp();
    if (!op || op->getNumResults() != 1 || op->getNumRegions() ||
        (!value.getType().isIndex() && !value.getType().isInteger(64)) ||
        !isMemoryEffectFree(op) || !isSpeculatable(op)) return false;
    if (!seen.insert(op).second) return true;
    if (seen.size() > 64 || !llvm::all_of(op->getOperands(), collect)) return false;
    addressOps.push_back(op);
    return true;
  };
  if (!collect(address)) return false;
  // The packet initializer alone does not prove that several logical rows
  // form one equally spaced rectangle (a flattened higher-rank View can have
  // a gap at a leading-axis boundary). Prove affine row dependence and an
  // exact unit column coefficient in pointer-width arithmetic.
  struct Coordinates { bool row = false; uint64_t column = 0; };
  DenseMap<Value, Coordinates> coordinates;
  coordinates[rows.getInductionVar()] = {true, 0};
  coordinates[columns.getInductionVar()] = {false, 1};
  for (Operation *op : addressOps) {
    Coordinates value;
    if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
      if (!cast.getIn().getType().isIndex() || !cast.getOut().getType().isInteger(64)) return false;
      value = coordinates.lookup(cast.getIn());
    } else if (isa<arith::AddIOp, arith::SubIOp>(op)) {
      auto a = coordinates.lookup(op->getOperand(0)), c = coordinates.lookup(op->getOperand(1));
      value = {a.row || c.row, isa<arith::AddIOp>(op) ? a.column + c.column : a.column - c.column};
    } else if (auto product = dyn_cast<arith::MulIOp>(op)) {
      auto a = coordinates.lookup(product.getLhs()), c = coordinates.lookup(product.getRhs());
      bool left = a.row || a.column, right = c.row || c.column;
      if (left && right) return false;
      value = left ? a : c;
      if (value.column) {
        APInt factor;
        if (!matchPattern(left ? product.getRhs() : product.getLhs(), m_ConstantInt(&factor))) return false;
        value.column *= factor.getZExtValue();
      }
    } else if (llvm::any_of(op->getOperands(), [&](Value input) {
                 auto dependency = coordinates.lookup(input);
                 return dependency.row || dependency.column;
               })) return false;
    coordinates[op->getResult(0)] = value;
  }
  if (coordinates.lookup(address).column != 1) return false;
  OpBuilder b(call);
  Location loc = call.getLoc();
  Value column = arith::ConstantIndexOp::create(b, loc, first % stride);
  if (reductionOffset) column = arith::AddIOp::create(b, loc, column, reductionOffset);
  auto atRow = [&](int64_t row) -> Value {
    IRMapping mapping;
    mapping.map(rows.getInductionVar(), arith::ConstantIndexOp::create(b, loc, row));
    mapping.map(columns.getInductionVar(), column);
    for (Operation *op : addressOps) b.clone(*op, mapping);
    return mapping.lookupOrDefault(address);
  };
  Value start = atRow(first / stride), next = atRow(first / stride + 1);
  Value rowStride = arith::SubIOp::create(b, loc, next, start);
  Value pointer = LLVM::GEPOp::create(b, loc, base.getType(), b.getI8Type(), base, ValueRange{start});
  Value tile = arith::ConstantIntOp::create(b, loc, geometry[2], 8);
  auto module = call->getParentOfType<ModuleOp>();
  auto function = module.lookupSymbol<func::FuncOp>("llvm.x86.tileloadd64");
  auto direct = func::CallOp::create(b, loc, function, ValueRange{tile, pointer, rowStride});
  direct->setAttr("weft.x86.amx_snapshot_load", geometry);
  call.erase();
  return true;
}

class ForwardAMXSnapshotTransfers
    : public PassWrapper<ForwardAMXSnapshotTransfers, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForwardAMXSnapshotTransfers)
  StringRef getArgument() const override { return "weft-x86-forward-amx-snapshots"; }
  StringRef getDescription() const override { return "Read proven snapshot rows directly into selected AMX tiles"; }
  void runOnOperation() override {
    SmallVector<func::CallOp> calls;
    getOperation().walk([&](func::CallOp call) { if (call->hasAttr("weft.x86.amx_load")) calls.push_back(call); });
    int64_t forwarded = 0;
    llvm::SmallPtrSet<Operation *, 8> initializers;
    DominanceInfo dominance(getOperation());
    for (auto call : calls) {
      auto geometry = call->getAttrOfType<DenseI64ArrayAttr>("weft.x86.amx_load");
      int64_t rows = geometry[0], bytes = geometry[1];
      Value root = allocation(call.getOperand(0));
      APInt offset, stride;
      Value reductionOffset;
      int64_t lastReductionOffset = 0;
      bool constantOffset = matchPattern(call.getOperand(1), m_ConstantInt(&offset));
      if (!constantOffset) {
        auto address = linear(call.getOperand(1));
        if (address.terms.size() != 1 || address.terms.begin()->second != 1) continue;
        auto iv = dyn_cast<BlockArgument>(address.terms.begin()->first);
        auto loop = iv ? dyn_cast<scf::ForOp>(iv.getOwner()->getParentOp()) : scf::ForOp();
        APInt low, high, step;
        if (!loop || loop.getInductionVar() != iv || !loop->hasAttr("weft.x86.amx_contraction") ||
            !matchPattern(loop.getLowerBound(), m_ConstantInt(&low)) || !low.isZero() ||
            !matchPattern(loop.getUpperBound(), m_ConstantInt(&high)) ||
            !matchPattern(loop.getStep(), m_ConstantInt(&step)) || high.getSExtValue() <= 0 ||
            step.getSExtValue() <= 0 || high.getSExtValue() > 4096) continue;
        reductionOffset = iv;
        lastReductionOffset = (high.getSExtValue() - 1) / step.getSExtValue() * step.getSExtValue();
        offset = APInt(64, address.constant);
      }
      if (!root || !cast<MemRefType>(root.getType()).getElementType().isInteger(8) ||
          rows < 2 || rows > 16 || bytes < 1 || bytes > 64 ||
          !matchPattern(call.getOperand(2), m_ConstantInt(&stride))) continue;
      vector::StoreOp store;
      bool initializedOnce = true;
      SmallVector<Value> aliases{root};
      while (!aliases.empty() && initializedOnce) {
        for (Operation *user : aliases.pop_back_val().getUsers()) {
          if (auto cast = dyn_cast<memref::CastOp>(user)) { aliases.push_back(cast.getResult()); continue; }
          if (auto candidate = dyn_cast<vector::StoreOp>(user)) {
            if (store) { initializedOnce = false; break; }
            store = candidate;
          } else if (!isa<memref::DimOp>(user) && !user->hasAttr("weft.x86.amx_load")) {
            initializedOnce = false; break;
          }
        }
      }
      APInt start;
      if (initializedOnce && store && forwardPacketSnapshot(call, store, offset.getSExtValue(),
              reductionOffset, lastReductionOffset, stride.getSExtValue(), dominance)) {
        ++forwarded;
        initializers.insert(store);
        continue;
      }
      if (!initializedOnce || !store || !dominance.dominates(store, call) || store.getIndices().size() != 1 ||
          !matchPattern(store.getIndices()[0], m_ConstantInt(&start))) continue;
      SmallVector<Lane> rowLoads;
      bool valid = true;
      for (int64_t row = 0; row < rows && valid; ++row) {
        int64_t first = offset.getSExtValue() - start.getSExtValue() + row * stride.getSExtValue();
        auto head = lane(store.getValueToStore(), first, call);
        if (!head || !canDelay(head->load, call)) { valid = false; break; }
        for (int64_t byte = 1; byte < bytes + lastReductionOffset; ++byte) {
          auto next = lane(store.getValueToStore(), first + byte, call);
          if (!next || next->load != head->load || next->index != head->index + byte) { valid = false; break; }
        }
        rowLoads.push_back(*head);
      }
      if (!valid) continue;
      Value base;
      SmallVector<Value> offsets;
      SmallVector<Linear> addresses;
      for (Lane row : rowLoads) {
        auto gep = row.load.getData().getDefiningOp<LLVM::GEPOp>();
        if (!gep || !gep.getElemType().isInteger(8) || gep.getRawConstantIndices().size() != 1 ||
            gep.getDynamicIndices().size() != 1 || (base && base != gep.getBase())) { valid = false; break; }
        auto argument = dyn_cast<BlockArgument>(gep.getBase());
        if (!argument || !isa<func::FuncOp>(argument.getOwner()->getParentOp())) { valid = false; break; }
        base = gep.getBase();
        Value offsetValue = gep.getDynamicIndices().front();
        if (!offsetValue.getType().isInteger(64)) { valid = false; break; }
        offsets.push_back(offsetValue);
        Linear address = linear(offsetValue); address.constant += row.index;
        addresses.push_back(address);
      }
      if (!valid) continue;
      Linear step = addresses[1]; step.add(addresses[0], uint64_t(-1));
      for (int64_t row = 2; row < rows; ++row) {
        Linear expected = addresses[0]; expected.add(step, row);
        if (!(expected == addresses[row])) { valid = false; break; }
      }
      if (!valid) continue;
      OpBuilder b(call); Location loc = call.getLoc();
      auto constant = [&](int64_t n) -> Value { return arith::ConstantIntOp::create(b, loc, n, 64); };
      Value first = arith::AddIOp::create(b, loc, offsets[0], constant(rowLoads[0].index));
      Value second = arith::AddIOp::create(b, loc, offsets[1], constant(rowLoads[1].index));
      Value rowStride = arith::SubIOp::create(b, loc, second, first);
      if (reductionOffset)
        first = arith::AddIOp::create(b, loc, first,
            arith::IndexCastOp::create(b, loc, b.getI64Type(), reductionOffset));
      Value pointer = LLVM::GEPOp::create(b, loc, base.getType(), b.getI8Type(), base, ValueRange{first});
      auto function = getOperation().lookupSymbol<func::FuncOp>("llvm.x86.tileloadd64");
      Value tile = arith::ConstantIntOp::create(b, loc, geometry[2], 8);
      auto direct = func::CallOp::create(b, loc, function, ValueRange{tile, pointer, rowStride});
      direct->setAttr("weft.x86.amx_snapshot_load", geometry);
      call.erase(); ++forwarded;
      initializers.insert(store);
    }
    for (Operation *initializer : initializers) {
      auto store = cast<vector::StoreOp>(initializer);
      SmallVector<Value> aliases{store.getBase()};
      bool unread = true;
      while (!aliases.empty() && unread)
        for (Operation *user : aliases.pop_back_val().getUsers()) {
          if (auto cast = dyn_cast<memref::CastOp>(user)) aliases.push_back(cast.getResult());
          else if (user != initializer && !isa<memref::DimOp>(user)) { unread = false; break; }
        }
      if (unread) store.erase();
    }
    if (forwarded) getOperation()->setAttr("weft.x86.forwarded_amx_snapshots", IntegerAttr::get(IntegerType::get(&getContext(), 64), forwarded));
  }
};

// Keep the selected tile and source snapshot. For a multi-row transfer, prime
// its current rows before independent work, leaving the tile load at its use.
// Smaller transfers can issue their load early. Explicit tile identities keep
// either schedule from crossing a use or definition of the same tile register.
class ScheduleAMXLoads : public PassWrapper<ScheduleAMXLoads, OperationPass<ModuleOp>> {
  bool crosses(Operation *op, int64_t tile) {
    if (op->getNumRegions()) return false;
    if (auto call = dyn_cast<func::CallOp>(op)) {
      StringRef name = call.getCallee();
      if (name != "llvm.x86.tilezero" && name != "llvm.x86.tileloadd64" &&
          name != "llvm.x86.tdpbssd" && name != "llvm.x86.tdpbsud" &&
          name != "llvm.x86.tdpbusd" && name != "llvm.x86.tdpbuud" &&
          name != "llvm.x86.tdpbf16ps") return false;
      unsigned tileOperands = name.starts_with("llvm.x86.tdp") ? 3 : 1;
      for (Value value : call.getOperands().take_front(tileOperands)) {
        APInt id;
        if (!matchPattern(value, m_ConstantInt(&id)) || id.getSExtValue() == tile) return false;
      }
      return true;
    }
    return preservesExternalMemory(op);
  }
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAMXLoads)
  StringRef getArgument() const override { return "weft-x86-schedule-amx-loads"; }
  StringRef getDescription() const override { return "Issue proven AMX snapshot reads before independent local work"; }
  void runOnOperation() override {
    SmallVector<func::CallOp> loads;
    getOperation().walk([&](func::CallOp call) {
      if (call->hasAttr("weft.x86.amx_snapshot_load")) loads.push_back(call);
    });
    int64_t scheduled = 0;
    for (auto load : loads) {
      APInt tile;
      if (!matchPattern(load.getOperand(0), m_ConstantInt(&tile))) continue;
      // These are compile-time search limits, not target register capacities.
      llvm::SmallPtrSet<Operation *, 32> dependencies, boundaries;
      SmallVector<Value> pending(load.getOperands());
      while (!pending.empty()) {
        Operation *op = pending.pop_back_val().getDefiningOp();
        if (!op || op->getBlock() != load->getBlock() || dependencies.contains(op) || boundaries.contains(op)) continue;
        bool scalar = llvm::all_of(op->getResultTypes(), [](Type type) {
          return type.isIntOrIndex() || isa<LLVM::LLVMPointerType>(type);
        });
        if (dependencies.size() >= 128 || op->getNumRegions() || !scalar ||
            !isMemoryEffectFree(op) || !isSpeculatable(op)) {
          boundaries.insert(op); continue;
        }
        dependencies.insert(op);
        llvm::append_range(pending, op->getOperands());
      }
      Operation *before = load;
      unsigned independent = 0, visited = 0;
      for (Operation *op = load->getPrevNode(); op && visited++ < 512; op = op->getPrevNode()) {
        if (boundaries.contains(op) || !crosses(op, tile.getSExtValue())) break;
        before = op;
        independent += !dependencies.contains(op);
      }
      if (independent < 8) continue;
      while (dependencies.contains(before)) before = before->getNextNode();
      SmallVector<Operation *> move;
      for (Operation *op = before; op != load; op = op->getNextNode())
        if (dependencies.contains(op)) move.push_back(op);
      for (Operation *op : move) op->moveBefore(before);
      auto geometry = load->getAttrOfType<DenseI64ArrayAttr>("weft.x86.amx_snapshot_load");
      APInt spacing;
      bool separateRows = !matchPattern(load.getOperand(2), m_ConstantInt(&spacing)) || spacing.getSExtValue() >= 64;
      if (geometry[0] >= 8 && geometry[1] >= 32 && separateRows) {
        // Hint only addresses already proven active by snapshot forwarding.
        // Leave independent work between these hints and the actual tile load;
        // issuing both together adds instructions without hiding the row waits.
        OpBuilder b(before);
        auto prefetch = getOperation().lookupSymbol<func::FuncOp>("llvm.prefetch.p0");
        if (!prefetch) {
          OpBuilder::InsertionGuard guard(b); b.setInsertionPointToEnd(getOperation().getBody());
          prefetch = func::FuncOp::create(b, load.getLoc(), "llvm.prefetch.p0",
              b.getFunctionType({LLVM::LLVMPointerType::get(&getContext()), b.getI32Type(), b.getI32Type(), b.getI32Type()}, {}));
          prefetch.setPrivate();
        }
        auto constant = [&](int64_t n, unsigned bits = 64) -> Value {
          return arith::ConstantIntOp::create(b, load.getLoc(), n, bits);
        };
        Value read = constant(0, 32), locality = constant(3, 32), data = constant(1, 32);
        for (int64_t row = 0; row < geometry[0]; ++row) {
          Value offset = arith::MulIOp::create(b, load.getLoc(), load.getOperand(2), constant(row));
          // The row may straddle cache lines. Hint its final valid byte so a
          // cached first line does not hide a still-missing trailing line.
          offset = arith::AddIOp::create(b, load.getLoc(), offset, constant(geometry[1] - 1));
          Value pointer = LLVM::GEPOp::create(b, load.getLoc(), load.getOperand(1).getType(), b.getI8Type(),
              load.getOperand(1), ValueRange{offset});
          func::CallOp::create(b, load.getLoc(), prefetch, ValueRange{pointer, read, locality, data});
        }
      } else load->moveBefore(before);
      ++scheduled;
    }
    if (scheduled) getOperation()->setAttr("weft.x86.scheduled_amx_loads",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), scheduled));
  }
};
}
std::unique_ptr<Pass> weft::x86::createAMXSnapshotTransferPass() {
  return std::make_unique<ForwardAMXSnapshotTransfers>();
}
std::unique_ptr<Pass> weft::x86::createAMXPackingReusePass() {
  return std::make_unique<ReuseAMXPacking>();
}
std::unique_ptr<Pass> weft::x86::createAMXLoadSchedulePass() {
  return std::make_unique<ScheduleAMXLoads>();
}
