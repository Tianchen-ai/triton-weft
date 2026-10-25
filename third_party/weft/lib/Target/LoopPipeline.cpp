#include "Weft/Target/LoopPipeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace weft;

namespace {
struct StageInstance { IRMapping values; };
} // namespace

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
      if (producers.contains(&op) == producers.contains(owner) && positions.lookup(owner) >= positions.lookup(&op))
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
  StageInstance prologue;
  prologue.values.map(loop.getInductionVar(), loop.getLowerBound());
  for (auto [argument, initial] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
    prologue.values.map(argument, initial);
  for (Operation *operation : schedule.producers) clone(b, operation, prologue.values);

  const unsigned carryCount = loop.getInitArgs().size();
  const unsigned slotWidth = unsigned(schedule.transportInduction) + schedule.transported.size();
  const unsigned distance = schedule.depth - 1;
  auto bindInvariants = [&](StageInstance &instance) {
    for (Operation *operation : schedule.invariants) {
      operation->walk([&](Operation *nested) {
        for (Value value : nested->getResults())
          instance.values.map(value, prologue.values.lookup(value));
      });
    }
  };
  PhysicalPipelineExpansion result;
  result.guard = guard;
  auto slot = [&](StageInstance &instance, Value coordinate) {
    SmallVector<Value> values;
    if (schedule.transportInduction) values.push_back(coordinate);
    for (Value value : schedule.transported) values.push_back(instance.values.lookup(value));
    return values;
  };
  auto bindConsumer = [&](StageInstance &instance, ValueRange incoming, ValueRange saved) {
    bindInvariants(instance);
    for (auto [argument, value] : llvm::zip_equal(loop.getRegionIterArgs(), incoming))
      instance.values.map(argument, value);
    if (schedule.transportInduction) {
      instance.values.map(loop.getInductionVar(), saved.front());
      saved = saved.drop_front();
    }
    for (auto [original, value] : llvm::zip_equal(schedule.transported, saved))
      instance.values.map(original, value);
  };
  auto outgoing = [&](StageInstance &instance) {
    SmallVector<Value> values;
    for (Value value : yield.getOperands()) values.push_back(instance.values.lookupOrDefault(value));
    return values;
  };
  auto drain = [&](ValueRange incoming, ArrayRef<SmallVector<Value>> saved) {
    SmallVector<Value> values(incoming);
    for (const auto &pending : saved) {
      StageInstance instance;
      bindConsumer(instance, values, pending);
      for (Operation *operation : schedule.consumers) clone(b, operation, instance.values);
      values = outgoing(instance);
    }
    return values;
  };
  // Each saved version belongs to one real source iteration. A guarded fill
  // never reads a nonexistent iteration, even when the trip count is less than
  // the selected lookahead. Short domains drain just their initialized slots.
  SmallVector<SmallVector<Value>> pending{slot(prologue, loop.getLowerBound())};
  std::function<SmallVector<Value>(Value)> fill = [&](Value coordinate) -> SmallVector<Value> {
    Value next = arith::AddIOp::create(b, loc, coordinate, loop.getStep());
    if (pending.size() < distance) {
      Value available = arith::CmpIOp::create(b, loc, predicate, next, loop.getUpperBound());
      auto more = scf::IfOp::create(b, loc, loop.getResultTypes(), available, true);
      b.setInsertionPointToStart(&more.getThenRegion().front());
      StageInstance instance;
      bindInvariants(instance);
      instance.values.map(loop.getInductionVar(), next);
      for (Operation *operation : schedule.producers)
        if (!schedule.isInvariant(operation)) clone(b, operation, instance.values);
      pending.push_back(slot(instance, next));
      auto filled = fill(next);
      scf::YieldOp::create(b, loc, filled);
      pending.pop_back();
      b.setInsertionPointToStart(&more.getElseRegion().front());
      scf::YieldOp::create(b, loc, drain(loop.getInitArgs(), pending));
      b.setInsertionPointAfter(more);
      return SmallVector<Value>(more.getResults());
    }

    SmallVector<Value> initial(loop.getInitArgs());
    for (const auto &saved : pending) llvm::append_range(initial, saved);
    auto steady = scf::ForOp::create(b, loc, next, loop.getUpperBound(), loop.getStep(), initial);
    copyLoopAttributes(loop, steady, false);
    steady->setAttr("weft.pipeline.expanded_depth", b.getI64IntegerAttr(distance + 1));
    steady->setAttr("weft.pipeline.version_values", b.getI64IntegerAttr(distance * slotWidth));
    if (!steady.getBody()->empty()) b.eraseOp(steady.getBody()->getTerminator());
    b.setInsertionPointToStart(steady.getBody());
    StageInstance future, current;
    bindInvariants(future);
    future.values.map(loop.getInductionVar(), steady.getInductionVar());
    bindConsumer(current, steady.getRegionIterArgs().take_front(carryCount),
                 steady.getRegionIterArgs().slice(carryCount, slotWidth));
    result.steady = steady;
    for (Operation *operation : schedule.ordered) {
      if (schedule.isInvariant(operation)) continue;
      StageInstance &instance = schedule.isProducer(operation) ? future : current;
      result.steadyOperations[operation] = clone(b, operation, instance.values);
    }
    SmallVector<Value> returned = outgoing(current);
    llvm::append_range(returned, steady.getRegionIterArgs().drop_front(carryCount + slotWidth));
    llvm::append_range(returned, slot(future, steady.getInductionVar()));
    scf::YieldOp::create(b, loc, returned);
    b.setInsertionPointAfter(steady);
    SmallVector<SmallVector<Value>> remaining;
    for (unsigned i = 0; i < distance; ++i)
      remaining.emplace_back(steady.getResults().slice(carryCount + i * slotWidth, slotWidth));
    return drain(steady.getResults().take_front(carryCount), remaining);
  };
  auto finalValues = fill(loop.getLowerBound());
  scf::YieldOp::create(b, loc, finalValues);
  b.setInsertionPointToStart(&guard.getElseRegion().front());
  scf::YieldOp::create(b, loc, loop.getInitArgs());
  // The caller owns replacement. Keeping the source alive here lets the
  // scheduler inspect a disposable expansion without dangling stage identities.
  return result;
}

