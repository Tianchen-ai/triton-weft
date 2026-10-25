#include "Weft/Target/LoopPipeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace weft;

namespace {
struct StageInstance {
  IRMapping values;
  unsigned age = 0;
  Value coordinate;
};
} // namespace

LogicalResult weft::distributePhysicalPipelineStages(
    PhysicalPipeline &schedule, PipelineDependencies dependencies,
    llvm::function_ref<bool(Operation *)> reads,
    llvm::function_ref<bool(Value)> transportable) {
  if (schedule.depth < 2 || schedule.depth > 4) return failure();
  auto body = schedule.loop.getBody();
  llvm::SmallPtrSet<Operation *, 32> fromRead;
  for (Operation *op : schedule.producers) {
    unsigned stage = 0;
    bool dependent = false;
    auto operands = dependencies(op);
    if (failed(operands)) return failure();
    for (Value value : *operands) {
      auto owner = value.getDefiningOp();
      owner = owner ? body->findAncestorOpInBlock(*owner) : nullptr;
      if (owner && fromRead.contains(owner)) {
        dependent = true;
        stage = std::max(stage, schedule.stage(owner) + 1);
      }
    }
    if (schedule.isInvariant(op)) stage = 0;
    else if (dependent || reads(op)) fromRead.insert(op);
    schedule.stages[op] = std::min(stage, schedule.depth - 2);
  }
  for (Operation *op : schedule.consumers) schedule.stages[op] = schedule.depth - 1;
  // Stage values only increase, within the fixed depth domain. Each edge is
  // revisited at most once per possible stage increase; no search is grown.
  bool changed = true;
  for (unsigned sweep = 0; changed && sweep <= schedule.depth * schedule.ordered.size(); ++sweep) {
    changed = false;
    for (Operation *op : schedule.ordered) {
      auto operands = dependencies(op);
      if (failed(operands)) return failure();
      for (Value value : *operands) {
        auto owner = value.getDefiningOp();
        owner = owner ? body->findAncestorOpInBlock(*owner) : nullptr;
        if (!owner || !schedule.stages.count(owner) || schedule.isInvariant(owner)) continue;
        unsigned required = std::max(schedule.stage(owner), schedule.stage(op));
        if (schedule.stage(op) < required) { schedule.stages[op] = required; changed = true; }
        if (!transportable(value) && schedule.stage(owner) < required) {
          schedule.stages[owner] = required;
          changed = true;
        }
      }
    }
  }
  if (changed) return schedule.loop.emitError("pipeline stage propagation exceeded its finite domain");
  schedule.producers.clear();
  schedule.consumers.clear();
  for (Operation *op : schedule.ordered)
    (schedule.stage(op) == schedule.depth - 1 ? schedule.consumers : schedule.producers).push_back(op);
  return success();
}

LogicalResult weft::verifyPhysicalPipeline(const PhysicalPipeline &schedule,
                                          PipelineDependencies dependencies) {
  auto loop = schedule.loop;
  auto body = loop.getBody();
  if (schedule.depth < 2 || schedule.depth > 4 || schedule.producers.empty() ||
      schedule.consumers.empty() || schedule.ordered.size() > 4096 ||
      schedule.ordered.size() != schedule.producers.size() + schedule.consumers.size())
    return loop.emitError("physical pipeline has an incomplete or unbounded stage domain");
  llvm::SmallPtrSet<Operation *, 32> producers(schedule.producers.begin(), schedule.producers.end());
  llvm::SmallPtrSet<Operation *, 32> consumers(schedule.consumers.begin(), schedule.consumers.end());
  llvm::SmallPtrSet<Operation *, 32> invariants(schedule.invariants.begin(), schedule.invariants.end());
  if (producers.size() != schedule.producers.size() || consumers.size() != schedule.consumers.size() ||
      invariants.size() != schedule.invariants.size() ||
      llvm::any_of(invariants, [&](Operation *op) { return !producers.contains(op); }))
    return loop.emitError("physical pipeline membership must be unique and invariants must be producers");
  DenseMap<Operation *, unsigned> positions;
  for (auto [position, op] : llvm::enumerate(schedule.ordered))
    if (op->getBlock() != body || op->hasTrait<OpTrait::IsTerminator>() ||
        producers.contains(op) == consumers.contains(op) || !positions.try_emplace(op, position).second)
      return loop.emitError("physical pipeline order must contain every body operation exactly once");
    else if (schedule.stage(op) >= schedule.depth ||
             (consumers.contains(op) != (schedule.stage(op) == schedule.depth - 1)) ||
             (invariants.contains(op) && schedule.stage(op) != 0))
      return op->emitError("physical pipeline stage conflicts with its recurrence or invariant ownership");
  unsigned edges = 0;
  for (Operation &op : body->without_terminator()) {
    if (!positions.contains(&op)) return op.emitError("operation is absent from the selected pipeline order");
    auto operands = dependencies(&op);
    if (failed(operands)) return failure();
    for (Value value : *operands) {
      if (++edges > 65536) return loop.emitError("physical pipeline dependency domain exceeds its finite bound");
      if (auto argument = dyn_cast<BlockArgument>(value); argument && argument.getOwner() == body) {
        if (invariants.contains(&op)) return op.emitError("pipeline invariant captures iteration state");
        if (producers.contains(&op) && argument != loop.getInductionVar())
          return op.emitError("pipeline producer requires an uncomputed numerical carry");
      }
      Operation *owner = value.getDefiningOp();
      if (!owner) continue;
      owner = body->findAncestorOpInBlock(*owner);
      if (!owner) continue;
      if (invariants.contains(&op) && !invariants.contains(owner))
        return op.emitError("pipeline invariant depends on an iteration-varying value");
      if (producers.contains(&op) && consumers.contains(owner))
        return op.emitError("pipeline producer depends on its consumer stage");
      if (!invariants.contains(owner) && schedule.stage(owner) > schedule.stage(&op))
        return op.emitError("pipeline use precedes its producer stage");
      if (schedule.stage(owner) == schedule.stage(&op) && positions.lookup(owner) >= positions.lookup(&op))
        return op.emitError("pipeline order uses a value before its same-stage definition");
    }
  }
  return success();
}

FailureOr<PhysicalPipelineExpansion> weft::expandPhysicalPipeline(
    const PhysicalPipeline &schedule, IRRewriter &b,
    PipelineClone clone, PipelineLoopAttributes copyLoopAttributes) {
  if (schedule.depth < 2 || schedule.depth > 4)
    return schedule.loop->emitError("physical pipeline requires depth two through four"), failure();
  auto loop = schedule.loop;
  Location loc = loop.getLoc();
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  b.setInsertionPoint(loop);
  auto predicate = schedule.descending ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ult;
  Value nonEmpty = arith::CmpIOp::create(b, loc, predicate, loop.getLowerBound(), loop.getUpperBound());
  auto guard = scf::IfOp::create(b, loc, loop.getResultTypes(), nonEmpty, true);
  copyLoopAttributes(loop, guard, true);

  b.setInsertionPointToStart(&guard.getThenRegion().front());
  const unsigned carryCount = loop.getInitArgs().size();
  const unsigned distance = schedule.depth - 1;
  IRMapping invariantValues;
  for (Operation *operation : schedule.invariants) clone(b, operation, invariantValues);
  auto bindInvariants = [&](StageInstance &instance) {
    for (Operation *operation : schedule.invariants) {
      operation->walk([&](Operation *nested) {
        for (Value value : nested->getResults())
          instance.values.map(value, invariantValues.lookup(value));
      });
    }
  };
  PhysicalPipelineExpansion result;
  result.guard = guard;

  // A source value is carried only across the stages that actually use it.
  // A depth-four schedule need not keep three versions of a value born at S2.
  SmallVector<SmallVector<Value>> frontiers(distance);
  SmallVector<bool> induction(distance, false);
  unsigned versionValues = 0;
  for (Value value : schedule.transported) {
    unsigned first = schedule.stage(value.getDefiningOp()), last = first;
    for (OpOperand &use : value.getUses()) {
      auto owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      if (owner == yield) last = distance;
      else if (owner) last = std::max(last, schedule.stage(owner));
    }
    for (unsigned age = first; age < last; ++age) frontiers[age].push_back(value);
  }
  for (OpOperand &use : loop.getInductionVar().getUses()) {
    auto owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
    unsigned stage = owner == yield ? distance : schedule.stage(owner);
    for (unsigned age = 0; age < stage; ++age) induction[age] = true;
  }
  for (unsigned age = 0; age < distance; ++age)
    versionValues += frontiers[age].size() + unsigned(induction[age]);

  auto slot = [&](StageInstance &instance) {
    SmallVector<Value> values;
    if (induction[instance.age]) values.push_back(instance.coordinate);
    for (Value value : frontiers[instance.age]) values.push_back(instance.values.lookup(value));
    return values;
  };
  auto restore = [&](ValueRange saved) {
    SmallVector<StageInstance> instances;
    unsigned cursor = 0;
    for (unsigned age = 0; age < distance; ++age) {
      StageInstance instance;
      instance.age = age;
      bindInvariants(instance);
      if (induction[age]) {
        instance.coordinate = saved[cursor++];
        instance.values.map(loop.getInductionVar(), instance.coordinate);
      }
      for (Value original : frontiers[age]) instance.values.map(original, saved[cursor++]);
      instances.push_back(std::move(instance));
    }
    assert(cursor == saved.size());
    return instances;
  };
  // Execute one physical time step. Every stage refers to its own real source
  // iteration; the last stage alone advances the ordered numerical recurrence.
  auto tick = [&](SmallVector<StageInstance> pending, Value coordinate,
                  SmallVector<Value> &carried, bool steady) {
    for (auto &instance : pending) ++instance.age;
    if (coordinate) {
      StageInstance future;
      future.coordinate = coordinate;
      bindInvariants(future);
      future.values.map(loop.getInductionVar(), coordinate);
      pending.insert(pending.begin(), std::move(future));
    }
    for (auto &instance : pending)
      if (instance.age == distance)
        for (auto [argument, value] : llvm::zip_equal(loop.getRegionIterArgs(), carried))
          instance.values.map(argument, value);
    for (Operation *operation : schedule.ordered) {
      if (schedule.isInvariant(operation)) continue;
      auto instance = llvm::find_if(pending, [&](const auto &entry) { return entry.age == schedule.stage(operation); });
      if (instance == pending.end()) continue;
      auto copy = clone(b, operation, instance->values);
      if (steady) result.steadyOperations[operation] = copy;
    }
    if (!pending.empty() && pending.back().age == distance) {
      carried.clear();
      for (Value value : yield.getOperands())
        carried.push_back(pending.back().values.lookupOrDefault(value));
      pending.pop_back();
    }
    return pending;
  };
  auto drain = [&](ValueRange incoming, SmallVector<StageInstance> pending) {
    SmallVector<Value> values(incoming);
    while (!pending.empty()) pending = tick(std::move(pending), {}, values, false);
    return values;
  };
  // Each saved version belongs to one real source iteration. A guarded fill
  // never reads a nonexistent iteration, even when the trip count is less than
  // the selected lookahead. Short domains drain just their initialized slots.
  SmallVector<Value> initialCarries(loop.getInitArgs());
  auto first = tick({}, loop.getLowerBound(), initialCarries, false);
  std::function<SmallVector<Value>(Value, SmallVector<StageInstance>)> fill =
      [&](Value coordinate, SmallVector<StageInstance> pending) -> SmallVector<Value> {
    Value next = arith::AddIOp::create(b, loc, coordinate, loop.getStep());
    if (pending.size() < distance) {
      Value available = arith::CmpIOp::create(b, loc, predicate, next, loop.getUpperBound());
      auto more = scf::IfOp::create(b, loc, loop.getResultTypes(), available, true);
      b.setInsertionPointToStart(&more.getThenRegion().front());
      SmallVector<Value> incoming(loop.getInitArgs());
      auto advanced = tick(pending, next, incoming, false);
      auto filled = fill(next, std::move(advanced));
      scf::YieldOp::create(b, loc, filled);
      b.setInsertionPointToStart(&more.getElseRegion().front());
      scf::YieldOp::create(b, loc, drain(loop.getInitArgs(), pending));
      b.setInsertionPointAfter(more);
      return SmallVector<Value>(more.getResults());
    }

    SmallVector<Value> initial(loop.getInitArgs());
    for (auto &instance : pending) llvm::append_range(initial, slot(instance));
    auto steady = scf::ForOp::create(b, loc, next, loop.getUpperBound(), loop.getStep(), initial);
    copyLoopAttributes(loop, steady, false);
    steady->setAttr("weft.pipeline.expanded_depth", b.getI64IntegerAttr(distance + 1));
    steady->setAttr("weft.pipeline.version_values", b.getI64IntegerAttr(versionValues));
    SmallVector<int64_t> stageOperations(schedule.depth, 0);
    for (Operation *operation : schedule.ordered)
      if (!schedule.isInvariant(operation)) ++stageOperations[schedule.stage(operation)];
    steady->setAttr("weft.pipeline.stage_operations", b.getDenseI64ArrayAttr(stageOperations));
    if (!steady.getBody()->empty()) b.eraseOp(steady.getBody()->getTerminator());
    b.setInsertionPointToStart(steady.getBody());
    result.steady = steady;
    SmallVector<Value> returned(steady.getRegionIterArgs().take_front(carryCount));
    auto advanced = tick(restore(steady.getRegionIterArgs().drop_front(carryCount)),
                         steady.getInductionVar(), returned, true);
    for (auto &instance : advanced) llvm::append_range(returned, slot(instance));
    scf::YieldOp::create(b, loc, returned);
    b.setInsertionPointAfter(steady);
    return drain(steady.getResults().take_front(carryCount), restore(steady.getResults().drop_front(carryCount)));
  };
  auto finalValues = fill(loop.getLowerBound(), std::move(first));
  scf::YieldOp::create(b, loc, finalValues);
  b.setInsertionPointToStart(&guard.getElseRegion().front());
  scf::YieldOp::create(b, loc, loop.getInitArgs());
  // The caller owns replacement. Keeping the source alive here lets the
  // scheduler inspect a disposable expansion without dangling stage identities.
  return result;
}

