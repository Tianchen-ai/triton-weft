#include "Weft/Target/X86Compiler.h"
#include "Resources.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
namespace {
// Array snapshots have already been materialized. Only closed numerical
// traversals may move; no external pointer access or author loop is a producer.
bool ownedConsumer(scf::ForOp loop) {
  return loop && (loop->hasAttr("weft.x86.array_pointwise") ||
                  loop->hasAttr("weft.x86.array_reduction"));
}
struct Access {
  Value buffer, value;
  SmallVector<Value> indices;
  bool write = false;
};
std::optional<Access> access(Operation *op) {
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return Access{load.getMemRef(), load.getResult(), llvm::to_vector(load.getIndices()), false};
  if (auto load = dyn_cast<vector::LoadOp>(op))
    return Access{load.getBase(), load.getResult(), llvm::to_vector(load.getIndices()), false};
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return Access{store.getMemRef(), store.getValue(), llvm::to_vector(store.getIndices()), true};
  if (auto store = dyn_cast<vector::StoreOp>(op))
    return Access{store.getBase(), store.getValueToStore(), llvm::to_vector(store.getIndices()), true};
  return std::nullopt;
}
Value readBuffer(Operation *op) {
  if (auto item = access(op); item && !item->write) return item->buffer;
  if (auto gather = dyn_cast<vector::GatherOp>(op)) return gather.getBase();
  return {};
}
Operation *ancestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block) op = op->getParentOp();
  return op;
}
bool sameValue(Value a, Value b) {
  if (a == b) return true;
  APInt x, y;
  return matchPattern(a, m_ConstantInt(&x)) && matchPattern(b, m_ConstantInt(&y)) && x == y;
}
bool sameDomain(scf::ForOp a, scf::ForOp b) {
  return sameValue(a.getLowerBound(), b.getLowerBound()) &&
         sameValue(a.getUpperBound(), b.getUpperBound()) && sameValue(a.getStep(), b.getStep());
}
bool linear(const Access &item, scf::ForOp loop) {
  return item.indices.size() == 1 && item.indices[0] == loop.getInductionVar();
}

// Fresh arrays cannot alias other allocations, and none of their references
// escape. Requiring every write to precede the producer also preserves a saved
// input snapshot across unrelated pointer writes, calls or lifecycle effects.
bool immutableInput(Value buffer, scf::ForOp producer) {
  auto alloc = buffer.getDefiningOp<memref::AllocOp>();
  if (!alloc || alloc->getBlock() != producer->getBlock()) return false;
  for (Operation *user : buffer.getUsers()) {
    if (readBuffer(user) == buffer) continue;
    auto item = access(user);
    if (!item || !item->write || item->buffer != buffer) return false;
    Operation *writer = ancestorInBlock(user, producer->getBlock());
    if (!writer || writer == producer.getOperation() || !writer->isBeforeInBlock(producer)) return false;
  }
  return true;
}
bool pureBody(scf::ForOp loop, llvm::SetVector<Value> &outputs) {
  if (!loop->hasAttr("weft.x86.array_pointwise") || loop.getNumResults()) return false;
  unsigned count = 0;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (++count > 64) return false;
    if (Value input = readBuffer(&op)) {
      if (!immutableInput(input, loop)) return false;
    } else if (auto item = access(&op); item && item->write) {
      auto alloc = item->buffer.getDefiningOp<memref::AllocOp>();
      if (!alloc || !alloc.getType().getLayout().isIdentity() ||
          alloc->getBlock() != loop->getBlock() || !linear(*item, loop)) return false;
      outputs.insert(item->buffer);
    } else {
      auto dialect = op.getName().getDialectNamespace();
      if (op.getNumRegions() || !isMemoryEffectFree(&op) ||
          (dialect != "arith" && dialect != "math" && dialect != "vector")) return false;
    }
  }
  return !outputs.empty();
}
struct Forwarding {
  scf::ForOp producer, consumer;
  SmallVector<Operation *> loads;
};
void recordFusion(scf::ForOp loop, OpBuilder &b, bool producer, unsigned bound = 0) {
  auto index = loop->getAttrOfType<IntegerAttr>("weft.x86.array_schedule_id");
  if (!index) return;
  auto function = loop->getParentOfType<func::FuncOp>();
  auto schedules = function->getAttrOfType<ArrayAttr>("weft.x86.array_schedule");
  SmallVector<Attribute> entries(schedules.begin(), schedules.end());
  NamedAttrList entry(cast<DictionaryAttr>(entries[index.getInt()]));
  if (producer) entry.set("kind", b.getStringAttr("forwarded-pointwise"));
  else {
    if (auto previous = dyn_cast_or_null<IntegerAttr>(entry.get("fusion_vector_bound")))
      bound = std::max(bound, unsigned(previous.getInt()));
    entry.set("fusion_vector_bound", b.getI64IntegerAttr(bound));
  }
  entries[index.getInt()] = entry.getDictionary(b.getContext());
  function->setAttr("weft.x86.array_schedule", b.getArrayAttr(entries));
}
bool forwardGroup(scf::ForOp seed, IRRewriter &b) {
  // Canonical birth/body blocks may have been flattened by SCF canonicalization.
  // Physical block identity alone does not preserve source lifecycle boundaries.
  auto scope = seed->getAttr("weft.x86.array_scope");
  if (!scope) return false;
  llvm::SetVector<Value> outputs;
  if (!pureBody(seed, outputs)) return false;
  llvm::SetVector<Operation *> writers;
  SmallVector<Operation *> loads;
  for (Value output : outputs) {
    llvm::SetVector<Operation *> localWriters;
    for (Operation *user : output.getUsers()) {
      auto item = access(user);
      if (!item || item->buffer != output) return false;
      auto loop = dyn_cast_or_null<scf::ForOp>(user->getParentOp());
      if (!ownedConsumer(loop) || loop->getBlock() != seed->getBlock() || !linear(*item, loop)) return false;
      if (loop->getAttr("weft.x86.array_scope") != scope) return false;
      if (item->write) localWriters.insert(loop);
      else loads.push_back(user);
    }
    if (localWriters.empty() || localWriters.size() > 2) return false;
    if (writers.empty()) writers = localWriters;
    else if (writers.size() != localWriters.size() ||
             llvm::any_of(localWriters, [&](Operation *op) { return !writers.contains(op); })) return false;
  }
  if (loads.empty()) return false;
  SmallVector<Forwarding> plans;
  for (Operation *op : writers) {
    auto writer = cast<scf::ForOp>(op);
    llvm::SetVector<Value> written;
    if (!pureBody(writer, written) || written.size() != outputs.size() ||
        llvm::any_of(written, [&](Value value) { return !outputs.contains(value); })) return false;
    plans.push_back({writer, {}, {}});
  }
  for (Operation *load : loads) {
    auto reader = cast<scf::ForOp>(load->getParentOp());
    auto item = *access(load);
    Forwarding *selected = nullptr;
    for (auto &plan : plans) {
      if (!sameDomain(plan.producer, reader)) continue;
      bool typedStore = llvm::any_of(plan.producer.getBody()->without_terminator(), [&](Operation &op) {
        auto store = access(&op);
        return store && store->write && store->buffer == item.buffer && store->value.getType() == item.value.getType();
      });
      if (!typedStore) continue;
      if (selected) return false;
      selected = &plan;
    }
    if (!selected || writers.contains(reader) || !selected->producer->isBeforeInBlock(reader)) return false;
    if (selected->consumer && selected->consumer != reader) return false;
    selected->consumer = reader; selected->loads.push_back(load);
  }
  auto function = seed->getParentOfType<func::FuncOp>();
  auto model = weft::x86::ResourceModel::get(seed);
  for (auto &plan : plans) {
    if (!plan.consumer || plan.loads.empty()) return false;
    if (std::distance(plan.consumer.getBody()->begin(), plan.consumer.getBody()->end()) > 128)
      return false;
  }
  SmallVector<scf::ForOp> candidates;
  SmallVector<weft::x86::LoopResources> resources;
  for (auto &plan : plans) {
    // Measure the proposed physical program, rather than adding two unrelated
    // lifetime bounds. Keep the original materialization until every partition fits.
    b.setInsertionPoint(plan.consumer);
    IRMapping cloned;
    auto candidate = cast<scf::ForOp>(b.clone(*plan.consumer, cloned));
    candidates.push_back(candidate);
    b.setInsertionPointToStart(candidate.getBody());
    IRMapping mapping; mapping.map(plan.producer.getInductionVar(), candidate.getInductionVar());
    DenseMap<Value, Value> forwarded;
    for (Operation &op : plan.producer.getBody()->without_terminator()) {
      if (auto item = access(&op); item && item->write)
        forwarded[item->buffer] = mapping.lookupOrDefault(item->value);
      else b.clone(op, mapping);
    }
    for (Operation *load : plan.loads) {
      auto item = *access(load);
      b.replaceOp(cloned.lookup(load), forwarded.at(item.buffer));
    }
    auto usage = weft::x86::analyzeLoop(candidate, model);
    if (!model.fits(usage.peak)) {
      weft::x86::updateArraySchedule(plan.consumer, b, {
        b.getNamedAttr("fusion_rejected_vectors", b.getI64IntegerAttr(usage.peak.vectors)),
        b.getNamedAttr("fusion_rejected_predicates", b.getI64IntegerAttr(usage.peak.predicates))});
      for (auto trial : candidates) b.eraseOp(trial);
      return false;
    }
    resources.push_back(usage);
  }
  for (auto [i, plan] : llvm::enumerate(plans)) {
    recordFusion(candidates[i], b, false, resources[i].peak.vectors);
    weft::x86::updateArraySchedule(candidates[i], b, weft::x86::resourceAttributes(b, resources[i]).getValue());
    recordFusion(plan.producer, b, true);
    b.replaceOp(plan.consumer, candidates[i].getResults());
  }
  for (auto &plan : plans) b.eraseOp(plan.producer);
  for (Value output : outputs) {
    assert(output.use_empty() && "all array users were closed before forwarding");
    b.eraseOp(output.getDefiningOp());
  }
  auto old = function->getAttrOfType<IntegerAttr>("weft.x86.forwarded_arrays");
  function->setAttr("weft.x86.forwarded_arrays", b.getI64IntegerAttr((old ? old.getInt() : 0) + outputs.size()));
  return true;
}
class ForwardArrays : public PassWrapper<ForwardArrays, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForwardArrays)
  StringRef getArgument() const override { return "weft-x86-forward-array-producers"; }
  StringRef getDescription() const override { return "Forward private numerical array producers into matching single-consumer traversals"; }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    bool changed;
    do {
      changed = false;
      SmallVector<scf::ForOp> loops;
      getOperation().walk([&](scf::ForOp loop) {
        if (loop->hasAttr("weft.x86.array_pointwise")) loops.push_back(loop);
      });
      for (auto loop : loops) if (forwardGroup(loop, b)) { changed = true; break; }
    } while (changed); // Every successful step removes at least one allocation.
  }
};
} // namespace
std::unique_ptr<Pass> weft::createX86ArrayForwardingPass() { return std::make_unique<ForwardArrays>(); }
