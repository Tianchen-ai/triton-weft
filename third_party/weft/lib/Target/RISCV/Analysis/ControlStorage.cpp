#include "Analysis/ControlStorage.h"
#include "Analysis/LocalMemory.h"
#include "Analysis/LocalLifetime.h"
#include "Transforms/Support/RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {

bool localNumeric(Type type) {
  auto value = dyn_cast<riscv::ValueType>(type);
  return value && value.getLayout().getCarrier() == "local" &&
      isa<IntegerType, FloatType>(value.getElementType());
}

// An edge owns a distinct destination for every logical slot, including when
// two slots initially contain the same SSA. Copies implement simultaneous phi
// assignment and preserve observations of the input before/after the control.
struct StorageSlot {
  unsigned position;
  Value storage;
};

struct ControlTransfer {
  Operation *terminator;
  SmallVector<Value> values;
  SmallVector<StorageSlot> slots;
  unsigned offset;
  SmallVector<unsigned> unchanged;
};

struct ControlFrame {
  Operation *operation;
  SmallVector<Value> incoming;
  SmallVector<StorageSlot> initial;
  SmallVector<ControlTransfer, 2> edges;
};

class ControlStorage {
public:
  ControlStorage(ModuleOp module, IRRewriter &b)
      : module(module), b(b), identity(lastLocalIdentity(module)) {}

  LogicalResult run() {
    // A local value can remain within its lexical scope while its old version
    // is observed after a later loop. Copy its incoming state once at the loop
    // boundary; updates then retain their ordinary versioned SSA handoff.
    SmallVector<Operation *> loops;
    module.walk<WalkOrder::PreOrder>([&](Operation *operation) {
      if (isa<scf::ForOp, scf::WhileOp>(operation)) loops.push_back(operation);
    });
    DominanceInfo dominance(module);
    for (Operation *operation : loops) {
      auto isolate = [&](OpOperand &operand, bool unchanged) -> LogicalResult {
        Value incoming = operand.get();
        if (unchanged || !localNumeric(incoming.getType())) return success();
        Operation *unproven = nullptr;
        if (hasExclusiveLocalVersion(incoming, operation, dominance, unproven)) return success();
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(operation);
        auto storage = allocateLocalValue(b, cast<riscv::ValueType>(incoming.getType()),
                                         operation, identity, false, "control-value");
        if (failed(storage) || failed(initializeLocalValue(b, *storage, incoming, operation)))
          return failure();
        operand.set(*storage);
        return success();
      };
      if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        for (auto [position, operand] : llvm::enumerate(loop.getInitArgsMutable()))
          if (failed(isolate(operand, loop.getYieldedValues()[position] ==
                                      loop.getRegionIterArgs()[position]))) return failure();
      } else if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
        auto condition = loop.getConditionOp();
        auto yielded = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
        for (auto [position, operand] : llvm::enumerate(loop.getInitsMutable())) {
          bool unchanged = position < condition.getArgs().size() &&
              condition.getArgs()[position] == loop.getBeforeBody()->getArgument(position) &&
              yielded.getOperand(position) == loop.getAfterBody()->getArgument(position);
          if (failed(isolate(operand, unchanged))) return failure();
        }
      }
    }
    // While's backedge and condition have independent signatures. Other
    // controls have a single set of joined slots.
    struct Escapes { llvm::BitVector joined, backedge; };
    llvm::DenseMap<Operation *, Escapes> selected;
    SmallVector<Value> pending;
    llvm::DenseSet<Value> scheduled;
    auto enqueue = [&](Value value) {
      if (scheduled.insert(value).second) pending.push_back(value);
    };
    module.walk([&](Operation *birth) {
      bool numericalBirth = false;
      if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(birth))
        numericalBirth = conversion.getConversion().getKind() == "local_store";
      if (auto state = dyn_cast<riscv::NewOp>(birth)) numericalBirth = state.getInitialized();
      if (auto staged = dyn_cast<riscv::MaterializeOp>(birth)) numericalBirth = true;
      if (!numericalBirth || birth->getNumResults() != 1 || !localNumeric(birth->getResult(0).getType()) ||
          riscv::storageAliasesRemainInScope(birth->getResult(0), birth->getBlock())) return;
      enqueue(birth->getResult(0));
    });
    // Different births often reach the same loop or branch. Discover their
    // union once; cycles terminate at already-visited SSA values.
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (isa<scf::YieldOp, scf::ConditionOp, riscv::YieldOp>(user)) {
          Operation *control = user->getParentOp();
          if (isa<scf::ForOp, scf::WhileOp, scf::IfOp, riscv::LoopOp>(control)) {
            if (localNumeric(value.getType())) {
              bool condition = isa<scf::ConditionOp>(user);
              unsigned position = use.getOperandNumber() - (condition ? 1 : 0);
              auto &escapes = selected[control];
              auto &slots = isa<scf::WhileOp>(control) && !condition ? escapes.backedge : escapes.joined;
              if (slots.size() <= position) slots.resize(position + 1);
              slots.set(position);
            }
            SmallVector<Value> outgoing;
            riscv::appendStorageAliases(use, outgoing);
            for (Value alias : outgoing) enqueue(alias);
            continue;
          }
        }
        SmallVector<Value> aliases;
        riscv::appendStorageAliases(use, aliases);
        for (Value alias : aliases) enqueue(alias);
        if (aliases.empty() && user->hasTrait<OpTrait::IsTerminator>())
          return user->emitError("local numerical representation has an unsupported non-control escape");
      }
    }
    SmallVector<Operation *> controls;
    // First close lexical escapes from outside in. Copies are emitted only
    // after every edge names its actual slot, so nested joins expose precise
    // storage provenance without temporarily escaping their private lifetime.
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (selected.contains(op)) controls.push_back(op);
    });
    SmallVector<ControlFrame, 4> frames;
    for (Operation *control : controls) {
      ControlFrame frame{control, {}, {}, {}};
      const auto &escapes = selected.find(control)->second;
      auto close = [&](Operation *terminator, ValueRange values, ArrayRef<StorageSlot> slots,
                       unsigned offset, ArrayRef<unsigned> unchanged = {}) {
        frame.edges.push_back({terminator, llvm::to_vector(values), llvm::to_vector(slots),
                               offset, llvm::to_vector(unchanged)});
        for (const StorageSlot &slot : slots)
          terminator->setOperand(offset + slot.position, slot.storage);
      };
      if (auto loop = dyn_cast<scf::ForOp>(control)) {
        auto slots = allocate(loop, escapes.joined, loop.getInitArgs().getTypes(), loop.getInitArgs());
        if (failed(slots)) return failure();
        frame.incoming = llvm::to_vector(loop.getInitArgs());
        frame.initial = *slots;
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        close(yield, yield.getOperands(), *slots, 0);
        for (const auto &slot : *slots) loop.getInitArgsMutable()[slot.position].set(slot.storage);
      } else if (auto loop = dyn_cast<riscv::LoopOp>(control)) {
        auto slots = allocate(loop, escapes.joined, loop.getCarried().getTypes(), loop.getCarried());
        if (failed(slots)) return failure();
        frame.incoming = llvm::to_vector(loop.getCarried());
        frame.initial = *slots;
        auto yield = cast<riscv::YieldOp>(loop.getBody().front().getTerminator());
        close(yield, yield.getValues(), *slots, 0);
        for (const auto &slot : *slots) loop.getCarriedMutable()[slot.position].set(slot.storage);
      } else if (auto branch = dyn_cast<scf::IfOp>(control)) {
        auto slots = allocate(branch, escapes.joined, branch.getResultTypes());
        if (failed(slots)) return failure();
        close(branch.thenYield(), branch.thenYield().getOperands(), *slots, 0);
        close(branch.elseYield(), branch.elseYield().getOperands(), *slots, 0);
      } else if (auto loop = dyn_cast<scf::WhileOp>(control)) {
        // before and after may have different signatures. Keep two explicit
        // sets of objects; condition(false) still supplies the loop results.
        auto before = allocate(loop, escapes.backedge, loop.getInits().getTypes(), loop.getInits());
        if (failed(before)) return failure();
        frame.incoming = llvm::to_vector(loop.getInits());
        frame.initial = *before;
        auto condition = cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());
        SmallVector<Value> forwarded(loop.getNumResults());
        SmallVector<unsigned> unchanged;
        for (auto [position, value] : llvm::enumerate(condition.getArgs())) {
          auto argument = dyn_cast<BlockArgument>(value);
          if (!argument || argument.getOwner() != loop.getBeforeBody()) continue;
          for (const auto &slot : *before)
            if (slot.position == argument.getArgNumber() && slot.storage.getType() == value.getType()) {
              forwarded[position] = slot.storage;
              unchanged.push_back(position);
            }
        }
        auto after = allocate(loop, escapes.joined, loop.getResultTypes(), {}, forwarded);
        if (failed(after)) return failure();
        auto yield = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
        close(condition, condition.getArgs(), *after, 1, unchanged);
        close(yield, yield.getOperands(), *before, 0);
        for (const auto &slot : *before) loop.getInitsMutable()[slot.position].set(slot.storage);
      }
      frames.push_back(std::move(frame));
    }
    // The records only retain replaced SSA operands during this rewrite. They
    // are consumed immediately into ordinary copy/control operations, not a
    // parallel representation or a plan carried to another pass.
    for (ControlFrame &frame : llvm::reverse(frames)) {
      if (failed(initialize(frame.operation, frame.incoming, frame.initial))) return failure();
      for (ControlTransfer &edge : frame.edges)
        if (failed(transfer(edge.terminator, edge.values, edge.slots, edge.offset, edge.unchanged))) return failure();
    }
    return success();
  }

private:
  ModuleOp module;
  IRRewriter &b;
  int64_t identity;

  FailureOr<SmallVector<StorageSlot>> allocate(Operation *owner, const llvm::BitVector &selected,
      TypeRange types, ValueRange incoming = {}, ArrayRef<Value> forwarded = {}) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(owner);
    SmallVector<StorageSlot> slots;
    for (auto [position, type] : llvm::enumerate(types)) {
      if (position >= selected.size() || !selected.test(position)) continue;
      auto value = cast<riscv::ValueType>(type);
      if (!forwarded.empty() && forwarded[position]) {
        slots.push_back({static_cast<unsigned>(position), forwarded[position]});
        continue;
      }
      if (!incoming.empty()) {
        auto state = incoming[position].getDefiningOp<riscv::NewOp>();
        // A fresh initialized state with exactly this control use has no old
        // observer or aliased second slot. Its own lifetime already encloses
        // the control; preserve its author birth instead of copying it.
        if (state && state.getResult().hasOneUse() && state.getInitialized() &&
            isa<IntegerType, FloatType>(state.getInitial().getType())) {
          slots.push_back({static_cast<unsigned>(position), incoming[position]});
          continue;
        }
      }
      auto storage = allocateLocalValue(b, value, owner, identity, false, "control-value");
      if (failed(storage)) return failure();
      slots.push_back({static_cast<unsigned>(position), *storage});
    }
    return slots;
  }

  LogicalResult copy(Value source, Value destination, Operation *anchor) {
    if (source.getType() != destination.getType() || !localNumeric(source.getType()))
      return anchor->emitError("local control edge changes its numerical slot type");
    auto type = cast<riscv::ValueType>(source.getType());
    auto input = localView(b, source, anchor), output = localView(b, destination, anchor);
    if (failed(input) || failed(output)) return anchor->emitError("local control edge has no complete descriptor");
    auto target = anchor->getParentOfType<riscv::KernelOp>().getTarget();
    int64_t bits = std::max<int64_t>(8, logicalBitWidth(type));
    int64_t width = target.getVlenBits() / bits;
    std::optional<std::pair<int64_t, int64_t>> issue;
    if (!type.getShape().empty() && bits >= 8 && !type.getElementType().isInteger(1) && width > 1)
      issue = {{type.getAxisIds().asArrayRef().back(), width}};
    return forEachValuePiece(b, type, anchor,
        [&](ArrayRef<Value> coordinates, int64_t axis, int64_t span) -> LogicalResult {
      auto piece = valuePieceType(b, type, axis, span, target);
      if (!piece) return failure();
      auto from = valueMemoryWindow(b, *input, type, coordinates, piece, anchor);
      auto to = valueMemoryWindow(b, *output, type, coordinates, piece, anchor);
      if (failed(from) || failed(to)) return failure();
      auto loaded = riscv::LoadOp::create(b, anchor->getLoc(), piece, *from, Value(),
          unassignedAccess(b), unselectedLeaf(b));
      auto stored = riscv::StoreOp::create(b, anchor->getLoc(), loaded, *to,
          unassignedAccess(b), unselectedLeaf(b), riscv::FieldStorePlanAttr());
      copyOrigin(anchor, loaded);
      copyOrigin(anchor, stored);
      return success();
    }, issue);
  }

  LogicalResult initialize(Operation *control, ValueRange incoming, ArrayRef<StorageSlot> slots) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(control);
    for (const auto &slot : slots)
      if (incoming[slot.position] != slot.storage &&
          failed(copy(incoming[slot.position], slot.storage, control))) return failure();
    return success();
  }

  LogicalResult transfer(Operation *terminator, ValueRange values,
                          ArrayRef<StorageSlot> slots, unsigned operandOffset,
                          ArrayRef<unsigned> unchanged = {}) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(terminator);
    struct Move { Value source; Value destination; bool fresh; };
    SmallVector<Move> pending;
    auto identical = [](Value source, Value destination) {
      if (source == destination) return true;
      if (source.getType() != destination.getType()) return false;
      // Only complete numerical objects participate in this parallel copy.
      // A reshape or arbitrary address projection needs its own coordinate
      // proof and must not become an identity merely by sharing an allocation.
      if (source.getDefiningOp<riscv::ReshapeOp>() || destination.getDefiningOp<riscv::ReshapeOp>()) return false;
      auto from = riscv::storageOrigins(source), to = riscv::storageOrigins(destination);
      return from.isOwnedLocal() && to.isOwnedLocal() && from.allocations.size() == 1 &&
          to.allocations.size() == 1 && from.allocations.front() == to.allocations.front();
    };
    for (const auto &slot : slots) {
      Value source = values[slot.position];
      if (llvm::is_contained(unchanged, slot.position) || identical(source, slot.storage)) continue;
      auto conversion = source.getDefiningOp<riscv::ConvertLayoutOp>();
      bool fresh = conversion && conversion.getConversion().getKind() == "local_store";
      if (auto state = source.getDefiningOp<riscv::NewOp>()) fresh |= state.getInitialized();
      pending.push_back({source, slot.storage, fresh});
    }
    // These are simultaneous phi assignments. A destination can be written
    // once no remaining source can observe its previous occupant. Chains need
    // no snapshots; a cycle saves one source, and duplicate uses share it.
    while (!pending.empty()) {
      auto ready = llvm::find_if(pending, [&](const Move &move) {
        return llvm::all_of(pending, [&](const Move &other) {
          return other.fresh || riscv::haveDisjointLocalStorage(other.source, move.destination);
        });
      });
      if (ready != pending.end()) {
        if (failed(copy(ready->source, ready->destination, terminator))) return failure();
        pending.erase(ready);
      } else {
        auto cyclic = llvm::find_if(pending, [](const Move &move) { return !move.fresh; });
        assert(cyclic != pending.end());
        Value source = cyclic->source;
        auto snapshot = allocateLocalValue(b, cast<riscv::ValueType>(source.getType()),
            terminator, identity, false, "control-transfer");
        if (failed(snapshot) || failed(copy(source, *snapshot, terminator))) return failure();
        for (Move &move : pending)
          if (move.source == source) { move.source = *snapshot; move.fresh = true; }
      }
    }
    for (const auto &slot : slots)
      terminator->setOperand(operandOffset + slot.position, slot.storage);
    return success();
  }
};
} // namespace

LogicalResult riscv_internal::materializeControlStorage(ModuleOp module, IRRewriter &builder) {
  return ControlStorage(module, builder).run();
}
