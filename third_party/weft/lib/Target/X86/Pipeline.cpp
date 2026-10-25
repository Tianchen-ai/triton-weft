#include "Weft/Target/X86Compiler.h"
#include "Weft/Target/LoopPipeline.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "Resources.h"
#include "StorageAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ModRef.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
namespace {
Operation *inBody(Operation *op, Block *body) {
  while (op && op->getBlock() != body) op = op->getParentOp();
  return op;
}

using weft::analysis::externalOperands;

bool numeric(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return !vector.isScalable() && vector.getElementType().isIntOrIndexOrFloat();
  return type.isIntOrIndexOrFloat();
}

// Intrinsic effects come from LLVM's instruction contracts. A spelling prefix
// or the fact that a helper preserves AMX state is not a purity proof.
bool pureIntrinsic(func::CallOp call, llvm::LLVMContext &context) {
  auto id = llvm::Intrinsic::lookupIntrinsicID(call.getCallee());
  if (id == llvm::Intrinsic::not_intrinsic) return false;
  auto attributes = llvm::Intrinsic::getFnAttributes(context, id);
  // Every call remains guarded by its original iteration/control predicate.
  // No extra invocation is speculated; require termination and freedom from
  // memory/synchronization/unwind effects, not LLVM's stronger Speculatable.
  return attributes.getMemoryEffects().doesNotAccessMemory() &&
      attributes.hasAttribute(llvm::Attribute::WillReturn) &&
      attributes.hasAttribute(llvm::Attribute::NoUnwind) &&
      attributes.hasAttribute(llvm::Attribute::NoSync) &&
      !attributes.hasAttribute(llvm::Attribute::Convergent) &&
      !attributes.hasAttribute(llvm::Attribute::NoDuplicate);
}

Operation *unsupportedEffect(Operation *operation, llvm::LLVMContext &context,
              const llvm::SmallPtrSetImpl<Operation *> &privateEffects) {
  if (auto call = dyn_cast<func::CallOp>(operation))
    return pureIntrinsic(call, context) ? nullptr : operation;
  if (!privateEffects.contains(operation)) {
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(operation)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      interface.getEffects(effects);
      if (llvm::any_of(effects, [](const auto &effect) {
            return !isa<MemoryEffects::Read>(effect.getEffect());
          })) return operation;
    } else if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
               !isMemoryEffectFree(operation)) return operation;
    if (auto load = dyn_cast<LLVM::LoadOp>(operation))
      if (load.getVolatile_() || load.getOrdering() != LLVM::AtomicOrdering::not_atomic) return operation;
  }
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        if (!nested.hasTrait<OpTrait::IsTerminator>())
          if (Operation *unsupported = unsupportedEffect(&nested, context, privateEffects))
            return unsupported;
  return nullptr;
}

FailureOr<weft::PhysicalPipeline> schedule(scf::ForOp loop, unsigned depth,
                                          llvm::LLVMContext &context) {
  weft::PhysicalPipeline result;
  result.loop = loop;
  result.depth = depth;
  APInt step, lower;
  if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || step.isNegative() || step.isZero() ||
      !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero())
    return loop.emitError("x86 pipeline requires a normalized finite Level traversal"), failure();
  if (!llvm::all_of(loop.getResultTypes(), numeric))
    return loop.emitError("x86 pipeline carry requires numeric SSA; storage ownership cannot escape its instance"), failure();

  llvm::SmallPtrSet<Operation *, 32> consumers, privateEffects;
  bool escaped = false;
  loop.walk([&](Operation *allocation) {
    if (!isa<memref::AllocOp, memref::AllocaOp>(allocation)) return;
    auto uses = weft::x86::privateStorageUses(allocation->getResult(0), loop);
    if (failed(uses)) { escaped = true; return; }
    privateEffects.insert(allocation);
    consumers.insert(inBody(allocation, loop.getBody()));
    for (Operation *user : uses->users) {
      if (!loop->isProperAncestor(user)) { escaped = true; continue; }
      privateEffects.insert(user);
      consumers.insert(inBody(user, loop.getBody()));
    }
  });
  if (escaped) return loop.emitError("x86 pipeline private storage escapes its original iteration"), failure();
  SmallVector<Operation *> operations;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (operations.size() == 4096)
      return loop.emitError("x86 pipeline exceeds its bounded physical operation domain"), failure();
    if (Operation *unsupported = unsupportedEffect(&op, context, privateEffects)) {
      auto diagnostic = unsupported->emitError("x86 pipeline cannot move across an external write or unknown effect: ");
      diagnostic << unsupported->getName();
      if (auto call = dyn_cast<func::CallOp>(unsupported)) diagnostic << " @" << call.getCallee();
      return failure();
    }
    operations.push_back(&op);
  }
  auto closeConsumers = [&] {
    for (Operation *op : operations)
      for (Value operand : externalOperands(op)) {
        auto arg = dyn_cast<BlockArgument>(operand);
        if ((arg && arg.getOwner() == loop.getBody() && arg != loop.getInductionVar()) ||
            consumers.contains(inBody(operand.getDefiningOp(), loop.getBody()))) consumers.insert(op);
      }
  };
  closeConsumers();
  auto yield = loop.getBody()->getTerminator();
  for (Value value : yield->getOperands())
    if (auto def = inBody(value.getDefiningOp(), loop.getBody())) consumers.insert(def);
  closeConsumers();
  for (Operation *op : operations) (consumers.contains(op) ? result.consumers : result.producers).push_back(op);
  if (result.producers.empty() || result.consumers.empty())
    return loop.emitError("x86 pipeline has no independent producer and numerical carry frontier"), failure();

  llvm::SmallPtrSet<Operation *, 32> invariants;
  for (Operation *op : result.producers) {
    bool invariant = isMemoryEffectFree(op) && !op->hasAttr("weft.x86.birth");
    for (Value operand : externalOperands(op)) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if (arg && arg.getOwner() == loop.getBody()) invariant = false;
      if (auto def = inBody(operand.getDefiningOp(), loop.getBody()); def && !invariants.contains(def)) invariant = false;
    }
    if (invariant) { invariants.insert(op); result.invariants.push_back(op); }
  }
  for (Operation *op : result.consumers)
    for (Value value : externalOperands(op)) result.transportInduction |= value == loop.getInductionVar();
  for (Value value : yield->getOperands()) result.transportInduction |= value == loop.getInductionVar();
  llvm::append_range(result.ordered, result.producers);
  llvm::append_range(result.ordered, result.consumers);
  if (failed(weft::distributePhysicalPipelineStages(result,
      [](Operation *op) -> FailureOr<SmallVector<Value>> { return externalOperands(op); },
      [](Operation *op) {
        bool reads = false;
        op->walk([&](Operation *nested) {
          if (auto effects = dyn_cast<MemoryEffectOpInterface>(nested)) {
            SmallVector<MemoryEffects::EffectInstance> instances;
            effects.getEffects(instances);
            reads |= llvm::any_of(instances, [](auto effect) { return isa<MemoryEffects::Read>(effect.getEffect()); });
          }
        });
        return reads;
      }, [](Value value) {
        // Keep mask construction with its use. Retaining several versions of
        // these cheap predicates spends the small AVX-512 k-register bank on
        // bookkeeping instead of the numerical supply being pipelined.
        auto vector = dyn_cast<VectorType>(value.getType());
        return numeric(value.getType()) && !(vector && vector.getElementType().isInteger(1));
      }))) return failure();
  for (Operation *op : result.producers) {
    if (result.isInvariant(op)) continue;
    for (Value value : op->getResults()) {
      bool crosses = llvm::any_of(value.getUses(), [&](OpOperand &use) {
        auto owner = inBody(use.getOwner(), loop.getBody());
        return owner == yield || (owner && result.stage(owner) > result.stage(op));
      });
      if (!crosses) continue;
      if (!numeric(value.getType()))
        return op->emitError("x86 pipeline frontier cannot carry an unversioned storage descriptor"), failure();
      result.transported.push_back(value);
    }
  }
  if (result.transported.empty())
    return loop.emitError("x86 pipeline has no iteration-varying numerical supply"), failure();
  return result;
}

constexpr StringLiteral scheduleDepth = "weft.x86.pipeline_depth";
constexpr StringLiteral scheduleStage = "weft.x86.pipeline_stage";
constexpr StringLiteral scheduleOrder = "weft.x86.pipeline_order";
constexpr StringLiteral scheduleInvariant = "weft.x86.pipeline_invariant";

// Reconstruct the selected frontier without making a second scheduling choice.
FailureOr<weft::PhysicalPipeline> readSchedule(scf::ForOp loop) {
  weft::PhysicalPipeline result;
  result.loop = loop;
  auto depth = loop->getAttrOfType<IntegerAttr>(scheduleDepth);
  if (!depth || depth.getInt() < 2 || depth.getInt() > 4)
    return loop.emitError("x86 pipeline requires a selected depth from two to four"), failure();
  result.depth = depth.getInt();
  DenseMap<int64_t, Operation *> positions;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto stage = op.getAttrOfType<IntegerAttr>(scheduleStage);
    auto order = op.getAttrOfType<IntegerAttr>(scheduleOrder);
    if (!stage || stage.getInt() < 0 || stage.getInt() >= result.depth ||
        !order || order.getInt() < 0 || !positions.try_emplace(order.getInt(), &op).second)
      return op.emitError("x86 pipeline requires a unique order and exact stage for every operation"), failure();
    result.stages[&op] = stage.getInt();
  }
  for (unsigned position = 0; position < positions.size(); ++position) {
    auto found = positions.find(position);
    if (found == positions.end()) return loop.emitError("x86 pipeline order is not a dense permutation"), failure();
    Operation *op = found->second;
    result.ordered.push_back(op);
    (result.stage(op) == result.depth - 1 ? result.consumers : result.producers).push_back(op);
    if (op->hasAttr(scheduleInvariant)) result.invariants.push_back(op);
  }
  for (Operation *op : result.consumers)
    for (Value value : externalOperands(op)) result.transportInduction |= value == loop.getInductionVar();
  auto yield = loop.getBody()->getTerminator();
  for (Value value : yield->getOperands()) result.transportInduction |= value == loop.getInductionVar();
  for (Operation *op : result.producers) {
    if (result.isInvariant(op)) continue;
    for (Value value : op->getResults())
      if (llvm::any_of(value.getUses(), [&](OpOperand &use) {
            auto owner = inBody(use.getOwner(), loop.getBody());
            return owner == yield || (owner && result.stage(owner) > result.stage(op));
          })) result.transported.push_back(value);
  }
  if (failed(weft::verifyPhysicalPipeline(result, [](Operation *op) -> FailureOr<SmallVector<Value>> {
        return externalOperands(op);
      }))) return failure();
  return result;
}

class ScheduleLevels : public PassWrapper<ScheduleLevels, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleLevels)
  explicit ScheduleLevels(unsigned depth) : depth(depth) {}
  StringRef getArgument() const override { return "weft-x86-schedule-levels"; }
  StringRef getDescription() const override { return "Select a dependency- and ownership-preserving pipeline frontier"; }
  void runOnOperation() override {
    if (depth == 1) return;
    llvm::LLVMContext context;
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (!loop->hasAttr("weft.x86.level") || loop.getInitArgs().empty()) return;
      bool nested = false;
      loop.getBody()->walk([&](scf::ForOp inner) { nested |= inner->hasAttr("weft.x86.level"); });
      if (!nested) loops.push_back(loop);
    });
    if (loops.empty()) {
      getOperation().emitError("requested x86 pipeline has no eligible innermost Level with numerical carries");
      signalPassFailure(); return;
    }
    IRRewriter b(&getContext());
    for (auto loop : loops) {
      auto selected = schedule(loop, depth, context);
      if (failed(selected)) { signalPassFailure(); return; }
      loop->setAttr(scheduleDepth, b.getI64IntegerAttr(depth));
      for (auto [position, op] : llvm::enumerate(selected->ordered)) {
        op->setAttr(scheduleStage, b.getI64IntegerAttr(selected->stage(op)));
        op->setAttr(scheduleOrder, b.getI64IntegerAttr(position));
        if (selected->isInvariant(op)) op->setAttr(scheduleInvariant, b.getUnitAttr());
      }
    }
  }
private:
  unsigned depth;
};

class PipelineLevels : public PassWrapper<PipelineLevels, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineLevels)
  StringRef getArgument() const override { return "weft-x86-pipeline-levels"; }
  StringRef getDescription() const override { return "Expand selected physical pipeline stages and recheck residency"; }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(scheduleDepth)) loops.push_back(loop);
    });
    IRRewriter b(&getContext());
    for (auto loop : loops) {
      auto selected = readSchedule(loop);
      if (failed(selected)) { signalPassFailure(); return; }
      unsigned depth = selected->depth;
      auto expanded = weft::expandPhysicalPipeline(*selected, b,
          [](IRRewriter &b, Operation *source, IRMapping &mapping) {
            auto clone = b.clone(*source, mapping);
            clone->removeAttr(scheduleStage);
            clone->removeAttr(scheduleOrder);
            clone->removeAttr(scheduleInvariant);
            return clone;
          }, [](scf::ForOp source, Operation *target, bool owner) {
            for (auto attr : source->getAttrs()) {
              if (attr.getName() == scheduleDepth) continue;
              if (attr.getName() == "weft.x86.level" && !owner) continue;
              target->setAttr(attr.getName(), attr.getValue());
            }
          });
      if (failed(expanded)) { signalPassFailure(); return; }
      b.replaceOp(loop, expanded->guard.getResults());
      auto function = expanded->guard->getParentOfType<func::FuncOp>();
      auto model = weft::x86::ResourceModel::get(expanded->steady);
      auto peak = weft::x86::functionRegisterPeak(function, model);
      if (!model.fits(peak)) {
        function.emitError("selected pipeline depth exceeds vector/predicate residency: ")
            << peak.vectors << '/' << model.vectorRegisters << ", "
            << peak.predicates << '/' << model.predicateRegisters;
        signalPassFailure(); return;
      }
      // Keep the decision inspectable after structured control lowers to CFG.
      function->setAttr("weft.pipeline.expanded_depth", b.getI64IntegerAttr(depth));
      SmallVector<Attribute> records;
      if (auto previous = function->getAttrOfType<ArrayAttr>("weft.x86.pipeline_schedule"))
        llvm::append_range(records, previous);
      records.push_back(b.getDictionaryAttr({
          b.getNamedAttr("depth", b.getI64IntegerAttr(depth)),
          b.getNamedAttr("saved_values", expanded->steady->getAttr("weft.pipeline.version_values")),
          b.getNamedAttr("stage_operations", expanded->steady->getAttr("weft.pipeline.stage_operations")),
          b.getNamedAttr("vector_peak", b.getI64IntegerAttr(peak.vectors)),
          b.getNamedAttr("predicate_peak", b.getI64IntegerAttr(peak.predicates))}));
      function->setAttr("weft.x86.pipeline_schedule", b.getArrayAttr(records));
    }
    if (failed(verify(getOperation()))) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86PipelineSchedulePass(unsigned depth) {
  return std::make_unique<ScheduleLevels>(depth);
}
std::unique_ptr<Pass> weft::createX86PipelinePass() {
  return std::make_unique<PipelineLevels>();
}
