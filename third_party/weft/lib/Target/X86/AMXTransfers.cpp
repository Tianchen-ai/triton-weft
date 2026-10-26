#include "AMXContractions.h"
#include "StorageAnalysis.h"
#include "Resources.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include <cmath>

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
  return weft::x86::isPrivateStorage(value);
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
  return weft::analysis::storagePreservedBetween(load, use,
      preservesExternalMemory, [](Operation *op) {
        using weft::analysis::StorageControl;
        if (isa<scf::IfOp>(op)) return StorageControl::Branch;
        if (isa<scf::ForOp>(op) && op->hasAttr("weft.x86.amx_contraction"))
          return StorageControl::Repeated;
        return StorageControl::Opaque;
      });
}

struct Packing {
  memref::AllocOp allocation;
  SmallVector<vector::StoreOp> stores;
  SmallVector<Operation *> initializerAliases;
  weft::analysis::StorageUses uses;
  uint64_t bytes;
};

std::optional<uint64_t> allocationBytes(memref::AllocOp alloc) {
  auto type = alloc.getType();
  if (type.getRank() != 1 || !type.getElementType().isIntOrFloat()) return {};
  int64_t count = type.hasStaticShape() ? type.getNumElements() : -1;
  APInt dynamic;
  if (count < 0 && alloc.getDynamicSizes().size() == 1 &&
      matchPattern(alloc.getDynamicSizes().front(), m_ConstantInt(&dynamic)))
    count = dynamic.getSExtValue();
  uint64_t elementBytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
  if (count <= 0 || uint64_t(count) > weft::x86::privateStorageBudget / elementBytes)
    return {};
  return uint64_t(count) * elementBytes;
}

// Address aliases may cross structured control, but every byte must still have
// exactly one initializer before all reads. Never infer a write coordinate from
// its allocation origin: only identity casts are admitted on initializer bases.
std::optional<Packing> closedPacking(memref::AllocOp alloc) {
  auto bytes = allocationBytes(alloc);
  auto uses = weft::x86::privateStorageUses(alloc);
  if (!bytes || failed(uses)) return {};
  Packing result{alloc, {}, {}, std::move(*uses), *bytes};
  SmallVector<Operation *> readers;
  llvm::SmallPtrSet<Operation *, 16> aliases;
  for (Operation *user : result.uses.users) {
    if (auto store = dyn_cast<vector::StoreOp>(user)) {
      if (!store->hasAttr("weft.x86.amx_pack") ||
          store->getBlock() != alloc->getBlock() || allocation(store.getBase()) != alloc)
        return {};
      result.stores.push_back(store);
      Value base = store.getBase();
      while (auto cast = base.getDefiningOp<memref::CastOp>()) {
        if (cast->getBlock() != alloc->getBlock()) return {};
        if (aliases.insert(cast).second) result.initializerAliases.push_back(cast);
        base = cast.getSource();
      }
    } else if (auto call = dyn_cast<func::CallOp>(user)) {
      if (!call->hasAttr("weft.x86.amx_load")) return {};
      readers.push_back(call);
    } else if (isa<memref::CastOp, ViewLikeOpInterface, scf::ForOp,
                   scf::WhileOp, scf::IfOp, scf::ConditionOp, scf::YieldOp,
                   arith::SelectOp>(user)) {
      // Their outgoing aliases were exhaustively visited by the shared query.
    } else if (auto memory = dyn_cast<MemoryEffectOpInterface>(user)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      memory.getEffects(effects);
      if (!llvm::all_of(effects, [](auto &effect) {
            return isa<MemoryEffects::Read>(effect.getEffect());
          })) return {};
      readers.push_back(user);
    } else return {};
  }
  if (result.stores.empty() || readers.empty()) return {};
  llvm::sort(result.stores, [](auto a, auto b) { return a->isBeforeInBlock(b); });
  llvm::sort(result.initializerAliases, [](auto a, auto b) { return a->isBeforeInBlock(b); });
  DominanceInfo dominance;
  for (Operation *reader : readers)
    if (!dominance.dominates(result.stores.back(), reader)) return {};
  uint64_t count = *bytes / llvm::divideCeil(alloc.getType().getElementTypeBitWidth(), 8u);
  llvm::SmallBitVector written(count, false);
  for (auto store : result.stores) {
    APInt offset;
    auto vector = store.getVectorType();
    if (store.getIndices().size() != 1 ||
        !matchPattern(store.getIndices()[0], m_ConstantInt(&offset)) || offset.isNegative() ||
        offset.getZExtValue() > count || uint64_t(vector.getNumElements()) > count - offset.getZExtValue())
      return {};
    for (int64_t i = 0; i < vector.getNumElements(); ++i) {
      uint64_t index = offset.getZExtValue() + i;
      if (written.test(index)) return {};
      written.set(index);
    }
  }
  if (!written.all()) return {};
  return result;
}

// Recompute after each move/merge. Extending one packing's lifetime is compared
// with all other packed operands in the function, including other contractions.
// Structured branches are conservatively linearized; nested users occupy their
// entire parent operation. Unknown or escaping storage never grants free space.
std::optional<uint64_t> packingPeak(Packing &candidate, Operation *before,
                                    Packing *replaces = nullptr) {
  auto function = candidate.allocation->getParentOfType<func::FuncOp>();
  DenseMap<Operation *, unsigned> begin, end;
  unsigned position = 0;
  function.walk<WalkOrder::PreOrder>([&](Operation *op) { begin[op] = position++; });
  function.walk<WalkOrder::PostOrder>([&](Operation *op) {
    end[op] = begin.lookup(op);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block) end[op] = std::max(end[op], end.lookup(&nested));
  });
  SmallVector<int64_t> changes(position + 1, 0);
  bool complete = true;
  llvm::SmallPtrSet<Operation *, 16> packedAllocations;
  function.walk([&](vector::StoreOp store) {
    if (!store->hasAttr("weft.x86.amx_pack")) return;
    Value root = allocation(store.getBase());
    auto alloc = root ? root.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
    if (!alloc) { complete = false; return; }
    packedAllocations.insert(alloc);
  });
  if (!packedAllocations.contains(candidate.allocation)) return {};
  for (Operation *operation : packedAllocations) {
    auto alloc = cast<memref::AllocOp>(operation);
    if (replaces && alloc == replaces->allocation) continue;
    auto bytes = allocationBytes(alloc);
    auto uses = weft::x86::privateStorageUses(alloc);
    if (!bytes || failed(uses)) return {};
    unsigned first = begin.lookup(alloc), last = first;
    SmallVector<Operation *> users(uses->users);
    if (alloc == candidate.allocation && replaces)
      llvm::append_range(users, replaces->uses.users);
    for (Operation *user : users) {
      Operation *anchor = alloc->getBlock()->findAncestorOpInBlock(*user);
      if (!anchor) return {};
      last = std::max(last, end.lookup(anchor));
    }
    if (alloc == candidate.allocation) first = begin.lookup(before);
    changes[first] += *bytes;
    changes[last + 1] -= *bytes;
  }
  if (!complete) return {};
  uint64_t peak = 0;
  int64_t live = 0;
  for (int64_t change : changes) {
    live += change;
    peak = std::max(peak, uint64_t(live));
  }
  return peak;
}

void relocatePacking(Packing &packing, Operation *destination,
                     ArrayRef<Operation *> producers,
                     const IRMapping *mapping = nullptr) {
  // Move one closed initializer in its original physical order. Grouping all
  // producers before all stores needlessly keeps every packet alive at once.
  // Preorder also preserves dependencies coming from enclosing blocks; the
  // movable-slice proof admits only pure operations without nested regions.
  llvm::SmallPtrSet<Operation *, 32> selected(producers.begin(), producers.end());
  selected.insert(packing.allocation);
  selected.insert(packing.initializerAliases.begin(), packing.initializerAliases.end());
  for (auto store : packing.stores) selected.insert(store);
  SmallVector<Operation *> ordered;
  packing.allocation->getParentOfType<func::FuncOp>().walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (selected.contains(op)) ordered.push_back(op);
  });
  for (Operation *op : ordered)
    (mapping ? mapping->lookup(op) : op)->moveBefore(destination);
}

weft::x86::RegisterUsage relocatedRegisters(Packing &packing, scf::ForOp destination,
    ArrayRef<Operation *> producers, const weft::x86::ResourceModel &model) {
  // Analyze the proposed SSA placement, including non-packing consumers of its
  // producers. An isolated initializer estimate misses values now live across
  // the consumer loop. The detached copy is physical IR, never a second plan.
  auto function = packing.allocation->getParentOfType<func::FuncOp>();
  IRMapping mapping;
  OwningOpRef<func::FuncOp> copy = cast<func::FuncOp>(function->clone(mapping));
  Operation *before = mapping.lookup(destination.getOperation());
  relocatePacking(packing, before, producers, &mapping);
  return weft::x86::functionRegisterPeak(*copy, model);
}

// Candidate scopes lie on one ancestor-loop chain. Invariant integer producers
// may start at different depths, so price each at its actual original frequency.
// Unknown loop counts remain symbolic: compare the leading steady-state term,
// retaining the constant startup term instead of inventing a runtime extent.
std::optional<SmallVector<double>> preparationSavings(Packing &packing,
    scf::ForOp destination, scf::ForOp scope, ArrayRef<Operation *> producers,
    const weft::x86::ResourceModel &model, double spillIssues) {
  auto after = weft::analysis::executionMultiplicity(destination, scope);
  if (!after) return {};
  SmallVector<Operation *> operations(producers);
  for (auto store : packing.stores) operations.push_back(store);
  SmallVector<double> savings;
  auto add = [&](const weft::analysis::ExecutionMultiplicity &weight, double cost) {
    if (!weight.fixed) return;
    size_t degree = weight.dynamicLoops.size();
    if (savings.size() <= degree) savings.resize(degree + 1, 0);
    savings[degree] += double(weight.fixed) * cost;
  };
  for (Operation *op : operations) {
    auto price = weft::x86::instructionCost(op, model);
    auto before = weft::analysis::executionMultiplicity(op, scope);
    if (!price || !before) return {};
    add(*before, price->issues); add(*after, -price->issues);
  }
  // LLVM may repair extra SIMD/predicate pressure. Charge an incremental native
  // spill/reload round trip per original preparation, rather than rejecting all
  // useful motion of a program whose wide SSA values already require allocation.
  auto preparation = weft::analysis::executionMultiplicity(packing.allocation, scope);
  if (!preparation) return {};
  add(*preparation, -spillIssues);
  if (!llvm::all_of(savings, [](double value) { return std::isfinite(value); })) return {};
  return savings;
}

int compareSavings(ArrayRef<double> left, ArrayRef<double> right) {
  for (size_t degree = std::max(left.size(), right.size()); degree > 0; --degree) {
    double a = degree <= left.size() ? left[degree - 1] : 0;
    double b = degree <= right.size() ? right[degree - 1] : 0;
    if (a != b) return a > b ? 1 : -1;
  }
  return 0;
}

// Packing a saved SSA snapshot is independent of a traversal that reuses that
// snapshot. Move the private materialization as a unit, including its exact
// integer/permutation producers, without moving or repeating the source read.
class ReuseAMXPacking : public PassWrapper<ReuseAMXPacking, OperationPass<ModuleOp>> {
  int64_t deduplicate() {
    SmallVector<memref::AllocOp> roots;
    llvm::SmallPtrSet<Operation *, 16> seen;
    getOperation().walk([&](vector::StoreOp store) {
      if (!store->hasAttr("weft.x86.amx_pack")) return;
      Value root = allocation(store.getBase());
      auto alloc = root ? root.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
      if (alloc && alloc->hasAttr("weft.x86.amx_pack_closed") && seen.insert(alloc).second) roots.push_back(alloc);
    });
    SmallVector<Packing, 4> retained;
    DominanceInfo dominance;
    int64_t merged = 0;
    for (auto alloc : roots) {
      auto closed = closedPacking(alloc);
      if (!closed) continue;
      auto &current = *closed;
      auto match = llvm::find_if(retained, [&](Packing &prior) {
        if (prior.allocation.getType() != alloc.getType() ||
            !dominance.dominates(prior.stores.back(), alloc) || prior.stores.size() != current.stores.size() ||
            prior.allocation.getDynamicSizes() != alloc.getDynamicSizes()) return false;
        for (auto [a, c] : llvm::zip(prior.stores, current.stores))
          if (a.getValueToStore() != c.getValueToStore() || a.getIndices() != c.getIndices()) return false;
        auto peak = packingPeak(prior, prior.allocation, &current);
        return peak && *peak <= weft::x86::privateStorageBudget;
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
    auto previous = getOperation()->getAttrOfType<IntegerAttr>("weft.x86.packing_passes");
    int64_t invocation = previous ? previous.getInt() + 1 : 1;
    getOperation()->setAttr("weft.x86.packing_passes", IntegerAttr::get(IntegerType::get(&getContext(),64),invocation));
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
      auto packing = closedPacking(alloc);
      if (!packing) continue;
      auto &stores = packing->stores;
      for (auto store : stores)
        for (StringRef name : {"weft.x86.amx_pack_resource_rejections", "weft.x86.amx_pack_candidates",
                               "weft.x86.amx_pack_residency"}) store->removeAttr(name);
      alloc->setAttr("weft.x86.amx_pack_closed", UnitAttr::get(&getContext()));
      struct Candidate {
        scf::ForOp loop;
        SmallVector<Operation *> producers;
        uint64_t peakBytes;
        unsigned depth;
      };
      SmallVector<Candidate> candidates;
      unsigned consideredDepth = 0, resourceRejected = 0;
      for (auto loop = alloc->getParentOfType<scf::ForOp>(); loop; loop = loop->getParentOfType<scf::ForOp>()) {
        if (++consideredDepth > 16) break;
        SmallVector<Value> roots(alloc.getDynamicSizes());
        for (auto store : stores) {
          roots.push_back(store.getValueToStore());
          llvm::append_range(roots, store.getIndices());
        }
        auto ordered = weft::analysis::movablePureSlice(roots, loop, [](Operation *op) {
          return llvm::all_of(op->getResultTypes(), [](Type type) { return getElementTypeOrSelf(type).isIntOrIndex(); });
        }, 4096);
        if (failed(ordered)) break;
        auto peak = packingPeak(*packing, loop);
        if (!peak || *peak > weft::x86::privateStorageBudget) {
          ++resourceRejected;
          break;
        }
        // A conditional region has no assumed execution probability. Keep the
        // packing in that region while still considering loops nested within it.
        if (!weft::analysis::executionMultiplicity(alloc, loop)) break;
        candidates.push_back({loop, std::move(*ordered), *peak, consideredDepth});
      }
      OpBuilder builder(&getContext());
      if (candidates.empty()) {
        if (resourceRejected)
          stores.front()->setAttr("weft.x86.amx_pack_resource_rejections", builder.getI64IntegerAttr(resourceRejected));
        continue;
      }
      auto model = weft::x86::ResourceModel::get(alloc);
      auto originalRegisters = weft::x86::functionRegisterPeak(alloc->getParentOfType<func::FuncOp>(), model);
      auto originalPeak = packingPeak(*packing, alloc);
      if (!originalPeak) continue;
      Candidate *chosen = nullptr;
      auto chosenRegisters = originalRegisters;
      uint64_t peakBytes = *originalPeak;
      SmallVector<double> chosenSavings;
      SmallVector<Attribute> records;
      unsigned selectedRecord = 0;
      for (auto &candidate : candidates) {
        auto registers = relocatedRegisters(*packing, candidate.loop, candidate.producers, model);
        auto incremental = [](uint64_t candidate, uint64_t original, uint64_t budget) {
          uint64_t existing = std::max(original, budget);
          return candidate > existing ? candidate - existing : uint64_t(0);
        };
        uint64_t vectorSpills = incremental(registers.vectors, originalRegisters.vectors, model.vectorRegisters);
        uint64_t predicateSpills = incremental(registers.predicates, originalRegisters.predicates, model.predicateRegisters);
        uint64_t spillBytes = vectorSpills * (model.vectorBits / 8) + predicateSpills * 8;
        double spillIssues = 2.0 * (vectorSpills + predicateSpills);
        auto saving = preparationSavings(*packing, candidate.loop, candidates.back().loop, candidate.producers, model, spillIssues);
        bool fits = spillBytes <= weft::x86::privateStorageBudget - candidate.peakBytes;
        if (!fits) ++resourceRejected;
        SmallVector<Attribute> coefficients;
        if (saving) for (double value : *saving) coefficients.push_back(builder.getF64FloatAttr(value));
        records.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("loops_crossed", builder.getI64IntegerAttr(candidate.depth)),
            builder.getNamedAttr("saved_issue_coefficients", builder.getArrayAttr(coefficients)),
            builder.getNamedAttr("packed_operand_peak_bytes", builder.getI64IntegerAttr(candidate.peakBytes)),
            builder.getNamedAttr("vector_peak", builder.getI64IntegerAttr(registers.vectors)),
            builder.getNamedAttr("predicate_peak", builder.getI64IntegerAttr(registers.predicates)),
            builder.getNamedAttr("incremental_spill_bytes", builder.getI64IntegerAttr(spillBytes)),
            builder.getNamedAttr("spill_reload_issue_units", builder.getF64FloatAttr(spillIssues)),
            builder.getNamedAttr("decision", builder.getStringAttr(!saving ? "unpriced-execution-scope" : !fits ? "resource-limit" : "legal"))}));
        if (!saving || !fits) continue;
        int comparison = compareSavings(*saving, chosenSavings);
        bool smaller = registers.vectors <= chosenRegisters.vectors && registers.predicates <= chosenRegisters.predicates &&
            candidate.peakBytes <= peakBytes && (registers.vectors < chosenRegisters.vectors ||
            registers.predicates < chosenRegisters.predicates || candidate.peakBytes < peakBytes);
        if (comparison > 0 || (comparison == 0 && smaller)) {
          chosen = &candidate; chosenSavings = *saving; chosenRegisters = registers;
          peakBytes = candidate.peakBytes; selectedRecord = records.size() - 1;
        }
      }
      if (chosen) {
        NamedAttrList selected(cast<DictionaryAttr>(records[selectedRecord]));
        selected.set("decision", builder.getStringAttr("selected"));
        records[selectedRecord] = selected.getDictionary(&getContext());
      }
      stores.front()->setAttr("weft.x86.amx_pack_candidates", builder.getArrayAttr(records));
      if (resourceRejected) stores.front()->setAttr("weft.x86.amx_pack_resource_rejections", builder.getI64IntegerAttr(resourceRejected));
      // Later transfer forwarding can eliminate the initializer itself. Keep
      // this pass-boundary decision alongside the other function schedules;
      // it is not evidence that a removed packing still executes at runtime.
      auto function = alloc->getParentOfType<func::FuncOp>();
      SmallVector<Attribute> schedules;
      if (auto prior = function->getAttrOfType<ArrayAttr>("weft.x86.packing_schedule")) llvm::append_range(schedules, prior);
      schedules.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("phase", builder.getStringAttr(getArgument())),
          builder.getNamedAttr("phase_invocation", builder.getI64IntegerAttr(invocation)),
          builder.getNamedAttr("candidates", builder.getArrayAttr(records)),
          builder.getNamedAttr("selected_candidate", builder.getI64IntegerAttr(chosen ? int64_t(selectedRecord) : -1)),
          builder.getNamedAttr("original_vector_peak", builder.getI64IntegerAttr(originalRegisters.vectors)),
          builder.getNamedAttr("original_predicate_peak", builder.getI64IntegerAttr(originalRegisters.predicates)),
          builder.getNamedAttr("packed_operand_bytes", builder.getI64IntegerAttr(packing->bytes))}));
      function->setAttr("weft.x86.packing_schedule", builder.getArrayAttr(schedules));
      if (!chosen) continue;
      auto destination = chosen->loop;
      auto &producers = chosen->producers;
      double preparationIssues = 0;
      unsigned unpriced = 0;
      for (Operation *producer : producers) {
        auto cost = weft::x86::instructionCost(producer, model);
        if (cost) preparationIssues += cost->issues;
        else ++unpriced;
      }
      for (auto store : stores) {
        auto cost = weft::x86::instructionCost(store, model);
        if (cost) preparationIssues += cost->issues;
        else ++unpriced;
      }
      auto residency = builder.getDictionaryAttr({
          builder.getNamedAttr("bytes", builder.getI64IntegerAttr(packing->bytes)),
          builder.getNamedAttr("packed_operand_peak_bytes", builder.getI64IntegerAttr(peakBytes)),
          builder.getNamedAttr("packed_operand_budget", builder.getI64IntegerAttr(weft::x86::privateStorageBudget)),
          builder.getNamedAttr("loops_crossed", builder.getI64IntegerAttr(chosen->depth)),
          builder.getNamedAttr("vector_peak", builder.getI64IntegerAttr(chosenRegisters.vectors)),
          builder.getNamedAttr("predicate_peak", builder.getI64IntegerAttr(chosenRegisters.predicates)),
          builder.getNamedAttr("original_vector_peak", builder.getI64IntegerAttr(originalRegisters.vectors)),
          builder.getNamedAttr("original_predicate_peak", builder.getI64IntegerAttr(originalRegisters.predicates)),
          builder.getNamedAttr("vector_budget", builder.getI64IntegerAttr(model.vectorRegisters)),
          builder.getNamedAttr("predicate_budget", builder.getI64IntegerAttr(model.predicateRegisters)),
          builder.getNamedAttr("cost_basis", builder.getStringAttr("complete-preparation-work; dynamic-loop-terms-retained")),
          builder.getNamedAttr("resource_rejections", builder.getI64IntegerAttr(resourceRejected)),
          builder.getNamedAttr("preparation_issue_units", builder.getF64FloatAttr(preparationIssues)),
          builder.getNamedAttr("unpriced_operations", builder.getI64IntegerAttr(unpriced))});
      // The first initializer survives allocation-to-stack-slot materialization.
      stores.front()->setAttr("weft.x86.amx_pack_residency", residency);
      relocatePacking(*packing, destination, producers);
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
