#include "LocalMemory.h"
#include "RISCVPhysicalSupport.h"
#include "DenseIssue.h"
#include "ValueLiveness.h"
#include "Weft/Dialect/RISCV/IR/LocalTransfer.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/DenseRelations.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

using namespace mlir;
using namespace weft;

namespace {

std::optional<int64_t> constantIndex(Value value) {
  return riscv::denseConstantIndex(value);
}

bool sameIndex(Value lhs, Value rhs) {
  if (riscv::sameDenseIndex(lhs, rhs)) return true;
  auto a = constantIndex(lhs), b = constantIndex(rhs);
  return a && b && a == b;
}

struct Rectangle {
  SmallVector<std::pair<int64_t, int64_t>> axes;
  int64_t elements = 1;
};

struct SymbolicInterval {
  Value begin, end;
  int64_t width;
};

std::optional<SymbolicInterval> symbolicCoverage(riscv::SubviewOp view, Operation *root,
                                                riscv::LocalViewOp destination) {
  auto shape = destination.getType().getShape().asArrayRef();
  if (llvm::count_if(shape, [](int64_t n) { return n < 0; }) != 1 ||
      view.getDynamicBounds().size() != shape.size() ||
      llvm::any_of(view.getOffsets(), [](int64_t offset) { return offset != -1; })) return {};
  scf::ForOp traversal;
  for (auto [axis, coordinate] : llvm::enumerate(view.getDynamicBounds())) {
    coordinate = riscv::denseIndexIdentity(coordinate);
    if (shape[axis] >= 0) {
      if (constantIndex(coordinate) != 0 || view.getExtents()[axis] != shape[axis]) return {};
      continue;
    }
    auto iv = dyn_cast<BlockArgument>(coordinate);
    traversal = iv ? dyn_cast<scf::ForOp>(iv.getOwner()->getParentOp()) : scf::ForOp();
    if (!traversal || traversal.getInductionVar() != iv ||
        (traversal != root && !root->isProperAncestor(traversal)) ||
        constantIndex(traversal.getStep()) != view.getExtents()[axis] || view.getExtents()[axis] <= 0) return {};
  }
  for (Operation *op = view; op && op != root->getParentOp(); op = op->getParentOp())
    if (auto loop = dyn_cast<scf::ForOp>(op); loop && loop != traversal &&
        analysis::staticTripCount(loop) != 1) return {};
  return SymbolicInterval{traversal.getLowerBound(), traversal.getUpperBound(),
                          *constantIndex(traversal.getStep())};
}

bool roundedExtent(Value value, Value extent, int64_t width) {
  if (width == 1) return sameIndex(value, extent);
  auto product = value.getDefiningOp<arith::MulIOp>();
  if (!product) return false;
  for (unsigned side : {0u, 1u}) {
    auto quotient = product->getOperand(side).getDefiningOp<arith::DivUIOp>();
    if (quotient && sameIndex(quotient.getLhs(), extent) &&
        constantIndex(quotient.getRhs()) == width && constantIndex(product->getOperand(1-side)) == width)
      return true;
  }
  return false;
}

bool completeSymbolicDomain(ArrayRef<SymbolicInterval> ranges, Value extent) {
  if (ranges.size() == 1)
    return ranges[0].width == 1 && constantIndex(ranges[0].begin) == 0 && sameIndex(ranges[0].end, extent);
  if (ranges.size() != 2) return false;
  for (unsigned first : {0u, 1u}) {
    const auto &main = ranges[first], &tail = ranges[1-first];
    if (constantIndex(main.begin) == 0 && roundedExtent(main.end, extent, main.width) &&
        tail.width == 1 && sameIndex(tail.begin, main.end) && sameIndex(tail.end, extent)) return true;
  }
  return false;
}

Value allocationSize(Value storage) {
  if (auto allocation = storage.getDefiningOp<riscv::LocalAllocOp>()) return allocation.getSizeBytes();
  if (auto reuse = storage.getDefiningOp<riscv::LocalReuseOp>()) return reuse.getSizeBytes();
  return {};
}

Value completeStorage(Value value) {
  while (auto slice = value.getDefiningOp<riscv::SliceOp>()) {
    if (!riscv::isCompleteDenseSlice(slice)) break;
    value = slice.getBase();
  }
  return value;
}

struct WriteTransfer {
  Operation *operation;
  Value input, destination;
};

bool completeExtents(ValueRange extents, riscv::ValueType type) {
  if (extents.size() != type.getShape().size()) return false;
  for (auto [extent, size] : llvm::zip(extents, type.getShape().asArrayRef()))
    if (constantIndex(extent) != size) return false;
  return true;
}

std::optional<WriteTransfer> writeTransfer(Operation *op) {
  if (auto store = dyn_cast<riscv::StoreOp>(op))
    return WriteTransfer{op, store.getValue(), store.getRegion()};
  if (auto store = dyn_cast<riscv::ScalarTileStoreOp>(op);
      store && !store.getGuarded() && completeExtents(store.getExtents(), store.getInput().getType()))
    return WriteTransfer{op, store.getInput(), store.getDestination()};
  if (auto store = dyn_cast<riscv::RVVTileStoreOp>(op);
      store && completeExtents(store.getExtents(), store.getInput().getType()))
    return WriteTransfer{op, store.getInput(), store.getDestination()};
  return std::nullopt;
}

Value readSource(Operation *op) {
  if (auto load = dyn_cast_or_null<riscv::LoadOp>(op); load && !load.getSnapshotStorage())
    return load.getRegion();
  if (auto load = dyn_cast_or_null<riscv::ScalarTileLoadOp>(op);
      load && !load.getGuarded() && completeExtents(load.getExtents(), load.getResult().getType()))
    return load.getSource();
  if (auto load = dyn_cast_or_null<riscv::RVVTileLoadOp>(op);
      load && completeExtents(load.getExtents(), load.getResult().getType()))
    return load.getSource();
  return {};
}

// A copy must cover each physical element exactly once. Checking the complete
// rectangular iteration domains admits vector strips plus scalar remainders,
// while rejecting partial initialization, repeated stores and sparse gathers.
std::optional<Rectangle> coverage(riscv::SubviewOp view, Operation *root) {
  if (view.getExtents().size() > 8 ||
      llvm::any_of(view.getExtents(), [](int64_t n) { return n <= 0; }) ||
      llvm::any_of(view.getOffsets(), [](int64_t n) { return n != -1; }) ||
      view.getDynamicBounds().size() != view.getExtents().size()) return std::nullopt;
  Rectangle result;
  llvm::DenseSet<Operation *> dimensions;
  for (auto [coordinate, width] : llvm::zip(view.getDynamicBounds(), view.getExtents())) {
    coordinate = riscv::denseIndexIdentity(coordinate);
    int64_t begin, end;
    if (auto constant = constantIndex(coordinate)) {
      begin = *constant;
      if (begin < 0 || width > std::numeric_limits<int64_t>::max() - begin) return std::nullopt;
      end = begin + width;
    } else {
      auto argument = dyn_cast<BlockArgument>(coordinate);
      auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp();
      if (!loop || loop.getInductionVar() != argument ||
          (loop != root && !root->isAncestor(loop)) || !dimensions.insert(loop).second) return std::nullopt;
      auto lower = constantIndex(loop.getLowerBound()), upper = constantIndex(loop.getUpperBound());
      auto step = constantIndex(loop.getStep());
      if (!lower || !upper || !step || *lower < 0 || *upper < *lower ||
          *step != width || (*upper - *lower) % width) return std::nullopt;
      begin = *lower;
      end = *upper;
    }
    result.axes.emplace_back(begin, end);
    if (end - begin && result.elements > std::numeric_limits<int64_t>::max() / (end - begin)) return std::nullopt;
    result.elements *= end - begin;
  }
  for (Operation *parent = view; parent && parent != root->getParentOp(); parent = parent->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForOp>(parent); loop && !dimensions.contains(loop)) {
      auto lower = constantIndex(loop.getLowerBound()), upper = constantIndex(loop.getUpperBound());
      auto step = constantIndex(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower || *upper - *lower > *step)
        return std::nullopt;
    }
    if (parent == root) break;
  }
  return result;
}

bool overlap(const Rectangle &lhs, const Rectangle &rhs) {
  if (!lhs.elements || !rhs.elements) return false;
  for (auto [a, b] : llvm::zip(lhs.axes, rhs.axes))
    if (a.second <= b.first || b.second <= a.first) return false;
  return true;
}

// A dead source can transfer ownership; a retained source needs a bounded byte
// transfer. Both choices consume this same complete coverage proof. Identify one
// completed traversal by its destination descriptor, not by a kernel name or
// the allocation's other writes in earlier/later control-flow iterations.
class CopyIssues {
public:
  explicit CopyIssues(riscv::LocalViewOp destination) : destination(destination) {}

  bool analyze() {
    auto target = destination->getParentOfType<riscv::KernelOp>().getTarget();
    SmallVector<Value> pending{destination.getResult()};
    llvm::DenseSet<Value> visited;
    for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
      if (cursor >= 256) return false;
      Value value = pending[cursor];
      if (!visited.insert(value).second) continue;
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (riscv::isStorageMetadataUse(use)) continue;
        if (auto slice = dyn_cast<riscv::SliceOp>(user); riscv::isCompleteDenseSlice(slice) && slice.getBase() == value) {
          pending.push_back(slice.getResult());
        } else if (auto view = dyn_cast<riscv::SubviewOp>(user); view && view.getBase() == value) {
          pending.push_back(view.getResult());
        } else if (auto store = writeTransfer(user); store && store->destination == value) {
          if (writes.size() >= 32) return false;
          writes.push_back(*store);
        } else if (readSource(user) == value) {
          reads.push_back(user);
        } else return false;
      }
    }
    if (writes.empty()) return false;
    bool dynamic = llvm::any_of(destination.getType().getShape().asArrayRef(), [](int64_t n) { return n < 0; });
    SmallVector<SymbolicInterval> symbolic;
    int64_t elements = 0;
    for (auto store : writes) {
      Operation *loaded = store.input.getDefiningOp();
      Value loadedSource = readSource(loaded);
      auto output = store.destination.getDefiningOp<riscv::SubviewOp>();
      auto input = loadedSource ? loadedSource.getDefiningOp<riscv::SubviewOp>() : riscv::SubviewOp();
      auto origin = input ? completeStorage(input.getBase()).getDefiningOp<riscv::LocalViewOp>() : riscv::LocalViewOp();
      if (!loadedSource || !store.input.hasOneUse() || !output || !input ||
          !origin || completeStorage(output.getBase()) != destination.getResult() ||
          input.getExtents() != output.getExtents() || input.getOffsets() != output.getOffsets() ||
          input.getDynamicBounds().size() != output.getDynamicBounds().size()) return false;
      for (auto [from, to] : llvm::zip(input.getDynamicBounds(), output.getDynamicBounds()))
        if (!sameIndex(from, to)) return false;
      if (source && source != origin) return false;
      source = origin;
      auto value = dyn_cast<riscv::ValueType>(store.input.getType());
      if (!value || value.getLayout().getValidity() != "full" ||
          value.getShape() != input.getType().getShape() ||
          value.getAxisIds() != input.getType().getAxisIds()) return false;
      auto count = riscv_internal::staticProduct(value.getShape().asArrayRef());
      if (!count || *count <= 0) return false;
      Operation *root = destination->getBlock()->findAncestorOpInBlock(*store.operation);
      if (!root || !destination->isBeforeInBlock(root)) return false;
      if (dynamic) {
        auto interval = symbolicCoverage(output, root, destination);
        if (!interval) return false;
        symbolic.push_back(*interval);
      } else {
        auto rectangle = coverage(output, root);
        if (!rectangle) return false;
        for (auto [range, extent] : llvm::zip(rectangle->axes, destination.getType().getShape().asArrayRef()))
          if (range.first < 0 || range.second > extent) return false;
        for (const auto &other : rectangles) if (overlap(*rectangle, other)) return false;
        if (rectangle->elements > std::numeric_limits<int64_t>::max() - elements ||
            rectangle->elements % *count) return false;
        elements += rectangle->elements;
        int64_t issues = rectangle->elements / *count;
        if (issues > std::numeric_limits<int64_t>::max() - originalIssues) return false;
        originalIssues += issues;
        rectangles.push_back(*rectangle);
      }
      if (!llvm::is_contained(roots, root)) roots.push_back(root);
      loads.insert(loaded);
    }
    if (dynamic) {
      auto a = source.getSource().getDefiningOp<riscv::LocalBindOp>();
      auto c = destination.getSource().getDefiningOp<riscv::LocalBindOp>();
      if (!a || !c || a.getType() != c.getType() || source.getType() != destination.getType() ||
          !riscv::haveDisjointLocalStorage(source, destination) ||
          !allocationSize(a.getStorage()) || !allocationSize(c.getStorage()) ||
          !sameIndex(allocationSize(a.getStorage()), allocationSize(c.getStorage()))) return false;
      for (auto [from, to] : llvm::zip(source.getExtents(), destination.getExtents()))
        if (!sameIndex(from, to)) return false;
      auto shape = destination.getType().getShape().asArrayRef();
      auto axis = llvm::find_if(shape, [](int64_t n) { return n < 0; }) - shape.begin();
      if (!completeSymbolicDomain(symbolic, destination.getExtents()[axis])) return false;
      bytes = -1;
    } else {
      auto total = riscv_internal::staticProduct(destination.getType().getShape().asArrayRef());
      auto extent = riscv::localTransferBytes(source.getResult(), destination.getResult());
      if (!total || elements != *total || !extent || !*extent) return false;
      bytes = *extent;
    }
    llvm::sort(roots, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    DominanceInfo dominance(destination->getParentOfType<riscv::KernelOp>());
    if (!dominance.properlyDominates(source.getOperation(), roots.front())) return false;
    // CSE may give later consumers the same descriptor as the initializer.
    // They are legal only after this copy's complete domain has been written.
    for (Operation *read : reads)
      if (!dominance.properlyDominates(roots.back(), read)) return false;
    for (Operation *root : roots) if (!copyOnly(root)) return false;
    // Preserve the original read and completion boundary. No intervening
    // access may change the source or observe a partially initialized target.
    for (Operation *op = roots.front(); op && op != roots.back(); op = op->getNextNode()) {
      if (llvm::is_contained(roots, op)) continue;
      if (op->getNumRegions()) return false;
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      if (!effects) { if (!isMemoryEffectFree(op)) return false; continue; }
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto effect : instances) {
        if (!isa<MemoryEffects::Read>(effect.getEffect()) || !effect.getValue() ||
            riscv::mayAliasStorage(effect.getValue(), destination.getResult())) return false;
      }
    }
    if (canReuse()) { reuse = true; return true; }
    if (canFillDestination()) { fillDestination = true; return true; }
    // Dynamic copies are eliminated only by the complete ownership proof. No
    // invented trip count is used to select a byte-transfer implementation.
    if (dynamic) return false;
    auto resident = riscv_internal::surroundingRegisterGroups(roots.front());
    if (!resident) return false;
    int64_t bestIssues = originalIssues;
    for (int64_t lmul : {8, 16, 32}) {
      if (!llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul) ||
          *resident > target.getVectorRegisters() - lmul / 8) continue;
      int64_t capacity = target.getVlenBits() * lmul / 64;
      int64_t issues = 1 + (bytes - 1) / capacity;
      if (issues < bestIssues) { bestIssues = issues; selectedLMUL = lmul; width = capacity; }
    }
    return selectedLMUL != 0;
  }

  LogicalResult rewrite(IRRewriter &b, llvm::DenseSet<Operation *> &erased) {
    Operation *anchor = roots.front();
    if (fillDestination) {
      // Produce into the already-owned destination slot. The original producer
      // and control path stay in place; only its unobservable scratch changes.
      b.replaceOp(filledSource, destination.getSource());
      if (filledAllocation->use_empty()) b.eraseOp(filledAllocation);
    } else if (reuse) {
      b.setInsertionPoint(replacedAllocation);
      auto transfer = riscv::LocalReuseOp::create(b, replacedAllocation.getLoc(), replacedAllocation.getType(),
          reusedSource.getStorage(), replacedAllocation.getSizeBytes());
      riscv_internal::copyOrigin(replacedAllocation, transfer);
      b.replaceOp(replacedAllocation, transfer.getResult());
    } else {
      b.setInsertionPoint(anchor);
      Value extent = arith::ConstantIndexOp::create(b, anchor->getLoc(), bytes);
      auto selected = riscv_internal::leaf(b, "transfer", "copy", "rvv.local-copy", "rvv.local-copy",
          0, 0, selectedLMUL / 8, 0, "none", "exact", {selectedLMUL, width});
      auto result = riscv_internal::emitDenseIssues(b, anchor, extent, width, {},
          [&](Value offset, Value active, bool, ValueRange) -> FailureOr<SmallVector<Value>> {
        auto copy = riscv::LocalCopyOp::create(b, anchor->getLoc(), source, destination, offset, active, selected);
        riscv_internal::copyOrigin(anchor, copy);
        return SmallVector<Value>{};
      });
      if (failed(result)) return anchor->emitError("selected local copy has no bounded issue traversal");
    }
    for (Operation *root : roots) {
      root->walk([&](Operation *operation) { erased.insert(operation); });
      b.eraseOp(root);
    }
    return success();
  }

private:
  riscv::LocalViewOp destination, source;
  SmallVector<WriteTransfer> writes;
  SmallVector<Operation *> reads;
  SmallVector<Rectangle> rectangles;
  SmallVector<Operation *> roots;
  llvm::DenseSet<Operation *> loads;
  int64_t bytes = 0, width = 0, selectedLMUL = 0, originalIssues = 0;
  bool reuse = false, fillDestination = false;
  riscv::LocalBindOp reusedSource;
  riscv::LocalAllocOp replacedAllocation;
  riscv::LocalBindOp filledSource;
  riscv::LocalAllocOp filledAllocation;

  bool canFillDestination() {
    filledSource = source.getSource().getDefiningOp<riscv::LocalBindOp>();
    auto target = destination.getSource().getDefiningOp<riscv::LocalBindOp>();
    if (!filledSource || !target || filledSource.getType() != target.getType()) return false;
    filledAllocation = filledSource.getStorage().getDefiningOp<riscv::LocalAllocOp>();
    if (!filledAllocation || !filledAllocation.getResult().hasOneUse()) return false;
    auto type = filledAllocation.getType(), to = target.getStorage().getType();
    if (type.getPurpose() != "pack" ||
        (type.getSchema() != "physical-share" && type.getSchema() != "local-value-copy") ||
        type.getSizeBytes() != bytes || to.getSizeBytes() != bytes ||
        type.getOwnerDomainId() != to.getOwnerDomainId() ||
        type.getLifetimeEndDomainId() != to.getLifetimeEndDomainId() ||
        type.getAlignment() > to.getAlignment()) return false;
    DominanceInfo dominance(destination->getParentOfType<riscv::KernelOp>());
    if (!dominance.properlyDominates(target.getOperation(), filledAllocation)) return false;
    Block *block = filledAllocation->getBlock();
    if (roots.front()->getBlock() != block || !filledAllocation->isBeforeInBlock(roots.front())) return false;
    SmallVector<Value> pending{filledSource.getResult()};
    llvm::DenseSet<Value> seen;
    for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
      if (cursor >= 1024) return false;
      Value value = pending[cursor];
      if (!seen.insert(value).second) continue;
      for (OpOperand &use : value.getUses()) {
        if (riscv::isStorageMetadataUse(use)) continue;
        Operation *user = use.getOwner();
        auto at = block->findAncestorOpInBlock(*user);
        if (!at) return false;
        if (!llvm::is_contained(roots, at) && !at->isBeforeInBlock(roots.front())) return false;
        riscv::appendStorageAliases(use, pending);
      }
    }
    // Earlier writes would become observable if the destination participated in
    // any intervening payload access. Check nested control/effects as well.
    for (Operation *at = filledAllocation; at && at != roots.back(); at = at->getNextNode()) {
      if (llvm::is_contained(roots, at)) continue;
      bool touched = false;
      at->walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::IsTerminator>() || riscv::isCoordinateIdentity(op)) return;
        if (auto memory = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> effects;
          memory.getEffects(effects);
          for (auto effect : effects) {
            if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
            Value storage = effect.getValue();
            if (!storage || riscv::mayAliasStorage(storage, target.getStorage())) touched = true;
          }
        } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() && !isMemoryEffectFree(op)) touched = true;
      });
      if (touched) return false;
    }
    return true;
  }

  bool canReuse() {
    auto binding = destination.getSource().getDefiningOp<riscv::LocalBindOp>();
    reusedSource = source.getSource().getDefiningOp<riscv::LocalBindOp>();
    if (!binding || !reusedSource || binding.getType() != reusedSource.getType()) return false;
    replacedAllocation = binding.getStorage().getDefiningOp<riscv::LocalAllocOp>();
    if (!replacedAllocation || !replacedAllocation.getResult().hasOneUse()) return false;
    auto type = binding.getStorage().getType(), previous = reusedSource.getStorage().getType();
    if ((type.getSchema() != "local-value-copy" && type.getSchema() != "physical-share" &&
         type.getSchema() != "staged-value" && type.getSchema() != "control-transfer" &&
         type.getSchema() != "control-value") ||
        type.getSizeBytes() != previous.getSizeBytes() || type.getSizeBytes() != bytes ||
        type.getElementType() != previous.getElementType() || type.getAlignment() > previous.getAlignment() ||
        type.getOwnerDomainId() != previous.getOwnerDomainId() ||
        type.getLifetimeEndDomainId() != previous.getLifetimeEndDomainId()) return false;
    Operation *birth = reusedSource.getStorage().getDefiningOp();
    if (!isa_and_nonnull<riscv::LocalAllocOp, riscv::LocalReuseOp>(birth) ||
        birth->getBlock() != replacedAllocation->getBlock() || !birth->isBeforeInBlock(replacedAllocation)) return false;
    for (Operation *root : roots)
      if (root->getBlock() != replacedAllocation->getBlock() || !replacedAllocation->isBeforeInBlock(root)) return false;
    // This descriptor names one traversal. Ownership transfer additionally
    // proves that every alias of the entire destination is initialized by it.
    SmallVector<Value> pending{binding.getResult()};
    llvm::DenseSet<Value> visited;
    for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
      if (cursor >= 1024) return false;
      Value value = pending[cursor];
      if (!visited.insert(value).second) continue;
      for (OpOperand &use : value.getUses()) {
        if (riscv::isStorageMetadataUse(use)) continue;
        SmallVector<Value> aliases;
        riscv::appendStorageAliases(use, aliases);
        Operation *user = use.getOwner();
        if (!aliases.empty() && (isMemoryEffectFree(user) ||
            isa<scf::ForOp, scf::WhileOp, scf::IfOp, riscv::LoopOp>(user))) {
          llvm::append_range(pending, aliases);
          continue;
        }
        llvm::append_range(pending, aliases);
        if (writeTransfer(user)) {
          if (!llvm::any_of(writes, [&](auto store) { return store.operation == user; })) {
            auto at = replacedAllocation->getBlock()->findAncestorOpInBlock(*user);
            if (!at || !roots.back()->isBeforeInBlock(at)) return false;
          }
          continue;
        }
        if (user->getNumRegions()) return false;
        if (!isMemoryEffectFree(user)) {
          auto memory = dyn_cast<MemoryEffectOpInterface>(user);
          if (!memory) return false;
          SmallVector<MemoryEffects::EffectInstance> effects;
          memory.getEffects(effects);
          if (llvm::any_of(effects, [](auto effect) {
                return !effect.getValue() || !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect());
              })) return false;
          auto at = replacedAllocation->getBlock()->findAncestorOpInBlock(*user);
          if (!at || !roots.back()->isBeforeInBlock(at)) return false;
        }
      }
    }
    pending.assign(1, reusedSource.getStorage());
    visited.clear();
    for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
      if (cursor >= 1024) return false;
      Value value = pending[cursor];
      if (!visited.insert(value).second) continue;
      for (OpOperand &use : value.getUses()) {
        if (riscv::isStorageMetadataUse(use)) continue;
        Operation *user = use.getOwner();
        if (isa<riscv::LocalReuseOp>(user)) return false;
        if (llvm::any_of(roots, [&](Operation *root) { return root == user || root->isAncestor(user); })) continue;
        SmallVector<Value> aliases;
        riscv::appendStorageAliases(use, aliases);
        if (!aliases.empty() && (isMemoryEffectFree(user) ||
            isa<scf::ForOp, scf::WhileOp, scf::IfOp, riscv::LoopOp>(user))) {
          // Constructing a descriptor does not observe its payload. Follow all
          // aliases to their real readers/writers before deciding whether the
          // old occupant survives this complete ownership transfer.
          llvm::append_range(pending, aliases);
          continue;
        }
        Operation *at = replacedAllocation->getBlock()->findAncestorOpInBlock(*user);
        if (!at || at == replacedAllocation || !at->isBeforeInBlock(replacedAllocation)) return false;
        llvm::append_range(pending, aliases);
      }
    }
    return true;
  }

  bool copyOnly(Operation *root) const {
    return !root->walk([&](Operation *op) {
      if (auto loop = dyn_cast<scf::ForOp>(op))
        return loop.getNumResults() == 0 ? WalkResult::advance() : WalkResult::interrupt();
      if (isa<scf::YieldOp>(op) || loads.contains(op)) return WalkResult::advance();
      if (writeTransfer(op))
        return llvm::any_of(writes, [&](auto store) { return store.operation == op; }) ?
            WalkResult::advance() : WalkResult::interrupt();
      if (op->getNumRegions() || !isMemoryEffectFree(op)) return WalkResult::interrupt();
      return WalkResult::advance();
    }).wasInterrupted();
  }
};
}

LogicalResult riscv_internal::forwardLocalCopies(ModuleOp module, IRRewriter &builder) {
  // Newly materialized copy traversals contain foldable index arithmetic.
  // Normalize it with MLIR's own folds before building coverage/alias queries;
  // an equivalent iv + 0 must not create an artificial coordinate mismatch.
  if (failed(applyPatternsGreedily(module, RewritePatternSet(module.getContext()))))
    return failure();
  SmallVector<riscv::LocalViewOp> views;
  module.walk([&](riscv::LocalViewOp view) { views.push_back(view); });
  llvm::DenseSet<Operation *> erased;
  for (auto view : views) {
    if (erased.contains(view)) continue;
    CopyIssues copy(view);
    if (copy.analyze()) {
      if (failed(copy.rewrite(builder, erased))) return failure();
    }
  }
  return success();
}
