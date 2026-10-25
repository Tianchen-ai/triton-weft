#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

using namespace mlir;

bool weft::analysis::isDefinedInside(Operation *operation, Value value) {
  Operation *owner = value.getDefiningOp();
  if (!owner) owner = value.getParentBlock()->getParentOp();
  return owner && (owner == operation || operation->isProperAncestor(owner));
}

SmallVector<Value> weft::analysis::externalOperands(Operation *operation) {
  SmallVector<Value> result;
  llvm::SmallPtrSet<Value, 32> seen;
  operation->walk([&](Operation *nested) {
    for (Value value : nested->getOperands())
      if (!isDefinedInside(operation, value) && seen.insert(value).second)
        result.push_back(value);
  });
  return result;
}

std::optional<uint64_t> weft::analysis::staticTripCount(Operation *operation) {
  auto loop = dyn_cast<scf::ForOp>(operation);
  APInt lower, upper, step;
  if (!loop || !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isStrictlyPositive() ||
      lower.getBitWidth() > 64 || upper.getBitWidth() > 64 || step.getBitWidth() > 64)
    return {};
  lower = loop.getUnsignedCmp() ? lower.zext(128) : lower.sext(128);
  upper = loop.getUnsignedCmp() ? upper.zext(128) : upper.sext(128);
  step = step.zext(128);
  if (upper.sle(lower)) return 0;
  APInt count = (upper - lower + step - 1).udiv(step);
  if (count.getActiveBits() > 64) return {};
  return count.getZExtValue();
}

std::optional<weft::analysis::ExecutionMultiplicity>
weft::analysis::executionMultiplicity(Operation *operation, Operation *scope) {
  if (operation == scope) return ExecutionMultiplicity{};
  if (!scope->isProperAncestor(operation)) return {};
  return executionMultiplicityInBlock(operation, scope->getBlock());
}

std::optional<weft::analysis::ExecutionMultiplicity>
weft::analysis::executionMultiplicityInBlock(Operation *operation, Block *scope,
                                             bool preserveConditionalDomain) {
  if (!scope || !scope->findAncestorOpInBlock(*operation)) return {};
  ExecutionMultiplicity result;
  SmallVector<Operation *> loops;
  for (Operation *current = operation; current; current = current->getParentOp()) {
    if (current->getBlock() == scope) {
      for (Operation *loop : loops) {
        auto count = staticTripCount(loop);
        if (count && !*count) return ExecutionMultiplicity{0, {}, {}};
        if (count && result.conditions.empty()) {
          if (result.fixed > std::numeric_limits<uint64_t>::max() / *count) return {};
          result.fixed *= *count;
        } else result.dynamicLoops.push_back(loop);
      }
      std::reverse(result.dynamicLoops.begin(), result.dynamicLoops.end());
      std::reverse(result.conditions.begin(), result.conditions.end());
      return result;
    }
    Operation *parent = current->getParentOp();
    if (!parent) return {};
    if (isa<scf::ForOp>(parent)) loops.push_back(parent);
    else if (auto branch = dyn_cast<scf::IfOp>(parent); branch && preserveConditionalDomain) {
      result.conditions.emplace_back(branch.getCondition(),
          current->getParentRegion() == &branch.getThenRegion());
    } else return {};
  }
  return {};
}

bool weft::analysis::isReadEpoch(Block &block,
                               llvm::function_ref<bool(Operation *)> ordinaryRead) {
  for (Operation &op : block) {
    if (op.getNumRegions()) return false;
    if (isMemoryEffectFree(&op)) continue;
    auto memory = dyn_cast<MemoryEffectOpInterface>(&op);
    if (!memory || !ordinaryRead(&op)) return false;
    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);
    if (!llvm::all_of(effects, [](const auto &effect) {
          return isa<MemoryEffects::Read>(effect.getEffect());
        })) return false;
  }
  return true;
}

FailureOr<SmallVector<Operation *>> weft::analysis::movablePureSlice(
    ValueRange values, Operation *anchor,
    llvm::function_ref<bool(Operation *)> accepts, unsigned limit) {
  DominanceInfo dominance;
  llvm::SmallPtrSet<Operation *, 32> seen;
  SmallVector<Operation *> ordered;
  auto visit = [&](auto &&self, Value value) -> bool {
    if (dominance.dominates(value, anchor)) return true;
    Operation *op = value.getDefiningOp();
    if (!op || op->getNumRegions() || op->hasTrait<OpTrait::IsTerminator>() ||
        !isMemoryEffectFree(op) || !isSpeculatable(op) || !accepts(op)) return false;
    if (seen.contains(op)) return true;
    if (seen.size() >= limit) return false;
    seen.insert(op);
    for (Value input : op->getOperands()) if (!self(self, input)) return false;
    ordered.push_back(op);
    return true;
  };
  for (Value value : values) if (!visit(visit, value)) return failure();
  return ordered;
}

weft::analysis::DependencyCost weft::analysis::dependencyCost(
    Block &block, ValueRange carries,
    llvm::function_ref<double(Operation *)> latency) {
  DenseMap<Value, double> paths, carried;
  for (Value value : carries) carried[value] = 0;
  DependencyCost result;
  for (Operation &op : block.without_terminator()) {
    double before = 0, fromCarry = -1;
    for (Value input : op.getOperands()) {
      before = std::max(before, paths.lookup(input));
      if (auto found = carried.find(input); found != carried.end())
        fromCarry = std::max(fromCarry, found->second);
    }
    double delay = latency(&op);
    result.criticalPath = std::max(result.criticalPath, before + delay);
    for (Value value : op.getResults()) {
      paths[value] = before + delay;
      if (fromCarry >= 0) carried[value] = fromCarry + delay;
    }
  }
  for (Value value : block.getTerminator()->getOperands())
    result.carriedPath = std::max(result.carriedPath, carried.lookup(value));
  return result;
}
