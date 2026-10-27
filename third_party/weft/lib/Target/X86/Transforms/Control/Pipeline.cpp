#include "Weft/Target/X86/X86Compiler.h"
#include "Weft/Transforms/LoopPipeline.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "Analysis/Resources.h"
#include "Analysis/StorageAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ModRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

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

constexpr StringLiteral bufferVersions = "weft.pipeline.buffer_versions";

SmallVector<Operation *> payloadOrder(scf::ForOp loop, const weft::analysis::StorageUses &uses) {
  llvm::SmallPtrSet<Operation *, 32> accesses;
  for (Operation *user : uses.users)
    if (!isMemoryEffectFree(user) && !isa<memref::AllocOp, memref::AllocaOp>(user))
      if (auto top = inBody(user, loop.getBody())) accesses.insert(top);
  SmallVector<Operation *> ordered;
  for (Operation &operation : loop.getBody()->without_terminator())
    if (accesses.contains(&operation)) ordered.push_back(&operation);
  return ordered;
}

std::optional<uint64_t> allocationBytes(Operation *allocation) {
  if (!isa<memref::AllocOp, memref::AllocaOp>(allocation)) return {};
  auto type = cast<MemRefType>(allocation->getResult(0).getType());
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() || !type.getElementType().isIntOrFloat()) return {};
  unsigned bits = type.getElementTypeBitWidth();
  if (isa<IntegerType>(type.getElementType())) bits = llvm::PowerOf2Ceil(std::max(8u, bits));
  uint64_t bytes = llvm::divideCeil(bits, 8u);
  for (int64_t extent : type.getShape()) {
    if (extent <= 0 || uint64_t(extent) > weft::x86::privateStorageBudget / bytes) return {};
    bytes *= extent;
  }
  return bytes;
}

// Only flat descriptor projections are rematerialized. Their non-address
// operands must be constants or values already dominating this loop, so slot
// selection never replays a numerical producer or a source-level birth.
std::optional<weft::PhysicalPipelineBuffer> bufferCandidate(
    Operation *allocation, scf::ForOp loop, const weft::analysis::StorageUses &uses) {
  auto bytes = allocationBytes(allocation);
  if (!bytes || allocation->getBlock() != loop.getBody() || allocation->hasAttr("weft.x86.birth")) return {};
  weft::PhysicalPipelineBuffer result;
  result.allocation = allocation;
  result.alignment = 64;
  if (auto alignment = allocation->getAttrOfType<IntegerAttr>("alignment"))
    result.alignment = std::max<uint64_t>(result.alignment, alignment.getInt());
  if (!llvm::isPowerOf2_64(result.alignment) || result.alignment > weft::x86::privateStorageBudget) return {};
  result.bytes = llvm::alignTo(*bytes, result.alignment);
  llvm::SmallPtrSet<Operation *, 32> descriptors;
  for (Value alias : uses.aliases) {
    if (alias == allocation->getResult(0)) continue;
    auto operation = alias.getDefiningOp();
    if (!operation) return {};
    if (operation->getBlock() != loop.getBody()) continue;
    if (!isa<memref::CastOp, ViewLikeOpInterface>(operation) || operation->getNumRegions()) return {};
    descriptors.insert(operation);
  }
  for (Operation *descriptor : descriptors)
    for (Value operand : descriptor->getOperands()) {
      if (operand == allocation->getResult(0)) continue;
      auto definition = inBody(operand.getDefiningOp(), loop.getBody());
      if (definition && !descriptors.contains(definition) && !isa<arith::ConstantOp>(definition)) return {};
      if (auto arg = dyn_cast<BlockArgument>(operand); arg && arg.getOwner() == loop.getBody()) return {};
    }
  for (Operation *user : uses.users) {
    auto top = inBody(user, loop.getBody());
    if (!top || top->hasTrait<OpTrait::IsTerminator>()) return {};
    if (top == user && isMemoryEffectFree(user) && !descriptors.contains(user)) return {};
  }
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (descriptors.contains(&operation)) result.descriptors.push_back(&operation);
  }
  result.accesses = payloadOrder(loop, uses);
  if (result.accesses.size() < 2) return {};
  return result;
}

void addBufferDependencies(weft::PhysicalPipeline &schedule,
                           ArrayRef<weft::PhysicalPipelineBuffer> buffers) {
  for (const auto &buffer : buffers)
    weft::addPipelineMemoryOrder(schedule, buffer.accesses);
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
  SmallVector<std::pair<Operation *, weft::analysis::StorageUses>> privateStorage;
  uint64_t occupied = 0;
  loop->getParentOfType<func::FuncOp>().walk([&](Operation *allocation) {
    if (auto stack = dyn_cast<LLVM::AllocaOp>(allocation)) {
      APInt elements;
      Type type = stack.getElemType();
      uint64_t count = 1;
      if (auto vector = dyn_cast<VectorType>(type)) { count = vector.getNumElements(); type = vector.getElementType(); }
      if (!type.isIntOrFloat() || !matchPattern(stack.getArraySize(), m_ConstantInt(&elements)) ||
          elements.isNegative() || elements.getActiveBits() > 32) {
        occupied = weft::x86::privateStorageBudget; return;
      }
      uint64_t width = llvm::divideCeil(type.getIntOrFloatBitWidth(), 8u);
      if (count > weft::x86::privateStorageBudget / width ||
          elements.getZExtValue() > (weft::x86::privateStorageBudget - occupied) / (count * width))
        occupied = weft::x86::privateStorageBudget;
      else occupied = std::min(weft::x86::privateStorageBudget,
          occupied + llvm::alignTo(count * width * elements.getZExtValue(), 64));
      return;
    }
    if (!isa<memref::AllocOp, memref::AllocaOp>(allocation)) return;
    auto bytes = allocationBytes(allocation);
    if (!bytes || *bytes > weft::x86::privateStorageBudget - occupied)
      occupied = weft::x86::privateStorageBudget;
    else occupied += llvm::alignTo(*bytes, 64);
    occupied = std::min(occupied, weft::x86::privateStorageBudget);
  });
  bool escaped = false;
  loop.walk([&](Operation *allocation) {
    if (!isa<memref::AllocOp, memref::AllocaOp>(allocation)) return;
    auto uses = weft::x86::privateStorageUses(allocation->getResult(0), loop);
    if (failed(uses)) { escaped = true; return; }
    privateEffects.insert(allocation);
    for (Operation *user : uses->users) {
      if (!loop->isProperAncestor(user)) { escaped = true; continue; }
      privateEffects.insert(user);
    }
    privateStorage.emplace_back(allocation, std::move(*uses));
  });
  if (escaped) return loop.emitError("x86 pipeline private storage escapes its original iteration"), failure();
  for (const auto &[allocation, uses] : privateStorage) {
    weft::addPipelineMemoryOrder(result, payloadOrder(loop, uses));
    auto candidate = bufferCandidate(allocation, loop, uses);
    if (candidate && candidate->bytes <= (weft::x86::privateStorageBudget - occupied) / (depth - 1)) {
      occupied += candidate->bytes * (depth - 1);
      result.buffers.push_back(std::move(*candidate));
    } else {
      consumers.insert(inBody(allocation, loop.getBody()));
      for (Operation *user : uses.users) consumers.insert(inBody(user, loop.getBody()));
    }
  }
  addBufferDependencies(result, result.buffers);
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
    for (Operation *op : operations) {
      for (Value operand : externalOperands(op)) {
        auto arg = dyn_cast<BlockArgument>(operand);
        if ((arg && arg.getOwner() == loop.getBody() && arg != loop.getInductionVar()) ||
            consumers.contains(inBody(operand.getDefiningOp(), loop.getBody()))) consumers.insert(op);
      }
      for (Operation *before : result.orderedAfter.lookup(op))
        if (consumers.contains(before)) consumers.insert(op);
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
      }, [&](Value value) {
        // Keep mask construction with its use. Retaining several versions of
        // these cheap predicates spends the small AVX-512 k-register bank on
        // bookkeeping instead of the numerical supply being pipelined.
        auto vector = dyn_cast<VectorType>(value.getType());
        return result.isBufferDescriptor(value.getDefiningOp()) ||
            (numeric(value.getType()) && !(vector && vector.getElementType().isInteger(1)));
      }))) return failure();
  weft::closePhysicalPipelineBuffers(result);
  for (Operation *op : result.producers) {
    if (result.isInvariant(op)) continue;
    for (Value value : op->getResults()) {
      if (result.isBufferDescriptor(op)) continue;
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
  if (result.transported.empty() && result.buffers.empty())
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
  bool invalidBuffers = false;
  loop.walk([&](Operation *allocation) {
    if (!isa<memref::AllocOp, memref::AllocaOp>(allocation)) return;
    auto uses = weft::x86::privateStorageUses(allocation->getResult(0), loop);
    if (failed(uses)) { allocation->emitError("pipeline private storage escaped its selected iteration"); invalidBuffers = true; return; }
    weft::addPipelineMemoryOrder(result, payloadOrder(loop, *uses));
    auto versions = allocation->getAttrOfType<IntegerAttr>(bufferVersions);
    if (!versions) return;
    auto buffer = bufferCandidate(allocation, loop, *uses);
    if (!buffer) { allocation->emitError("selected pipeline buffer lost its private descriptor closure"); invalidBuffers = true; return; }
    buffer->firstStage = result.depth;
    for (Operation *access : buffer->accesses) {
      buffer->firstStage = std::min(buffer->firstStage, result.stage(access));
      buffer->lastStage = std::max(buffer->lastStage, result.stage(access));
    }
    if (versions.getInt() != buffer->versions() || result.stage(allocation) != buffer->firstStage ||
        llvm::any_of(buffer->descriptors, [&](Operation *op) { return result.stage(op) != buffer->firstStage; })) {
      allocation->emitError("selected buffer versions disagree with its actual stage lifetime"); invalidBuffers = true; return;
    }
    result.buffers.push_back(std::move(*buffer));
  });
  if (invalidBuffers) return failure();
  addBufferDependencies(result, result.buffers);
  for (Operation *op : result.consumers)
    for (Value value : externalOperands(op)) result.transportInduction |= value == loop.getInductionVar();
  auto yield = loop.getBody()->getTerminator();
  for (Value value : yield->getOperands()) result.transportInduction |= value == loop.getInductionVar();
  for (Operation *op : result.producers) {
    if (result.isInvariant(op)) continue;
    for (Value value : op->getResults())
      if (!result.isBufferDescriptor(op) && llvm::any_of(value.getUses(), [&](OpOperand &use) {
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
      for (const auto &buffer : selected->buffers)
        buffer.allocation->setAttr(bufferVersions, b.getI64IntegerAttr(buffer.versions()));
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
          }, [](IRRewriter &b, const weft::PhysicalPipelineBuffer &buffer) -> FailureOr<Value> {
            auto oldType = cast<MemRefType>(buffer.allocation->getResult(0).getType());
            auto type = MemRefType::get({int64_t(buffer.bytes * buffer.versions())}, b.getI8Type(), MemRefLayoutAttrInterface{}, oldType.getMemorySpace());
            auto allocation = memref::AllocaOp::create(b, buffer.allocation->getLoc(), type);
            allocation.setAlignment(buffer.alignment);
            allocation->setAttr("weft.pipeline.buffer_slots", b.getI64IntegerAttr(buffer.versions()));
            return allocation.getResult();
          }, [](IRRewriter &b, const weft::PhysicalPipelineBuffer &buffer, Value storage, Value slot) -> FailureOr<Value> {
            auto loc = buffer.allocation->getLoc();
            Value bytes = arith::ConstantIndexOp::create(b, loc, buffer.bytes);
            Value offset = b.createOrFold<arith::MulIOp>(loc, slot, bytes);
            return memref::ViewOp::create(b, loc, buffer.allocation->getResult(0).getType(), storage, offset, ValueRange{}).getResult();
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
          b.getNamedAttr("buffer_versions", expanded->steady->getAttr("weft.pipeline.buffer_versions")),
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
