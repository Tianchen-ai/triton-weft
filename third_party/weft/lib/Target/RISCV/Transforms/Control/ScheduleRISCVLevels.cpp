#include "Weft/Target/RISCV/RISCVPasses.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Transforms/Control/PipelineSchedule.h"
#include "Transforms/Control/PipelineShapeBindings.h"
#include "Transforms/Support/RISCVPhysicalSupport.h"
#include "Analysis/ValueLiveness.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
// Evaluate the actual expanded SSA in a disposable kernel. The selected result
// remains stage/order on the original loop; only the expander installs a new
// program. This prevents scheduler and emitter from owning different pipelines.
LogicalResult selectFrontiers(PipelineSchedule &schedule, IRRewriter &b,
                             unsigned &interleaved, int64_t &largestReduction) {
  schedule.write();
  if (failed(verifyPipelineShapeStages(schedule.loop))) return failure();
  auto kernel = schedule.loop->getParentOfType<riscv::KernelOp>();
  IRMapping mapping;
  OwningOpRef<riscv::KernelOp> trial(cast<riscv::KernelOp>(kernel->clone(mapping)));
  auto loop = cast<scf::ForOp>(mapping.lookup(schedule.loop.getInductionVar()).getParentBlock()->getParentOp());
  auto copy = PipelineSchedule::read(loop);
  if (failed(copy)) return failure();
  int64_t identity = nextPipelineShapeIdentity(trial->getOperation());
  auto expanded = expandPipeline(*copy, b, identity);
  if (failed(expanded)) return failure();
  auto before = registerPeak(expanded->guard.getThenRegion().front());
  if (!before) return schedule.loop.emitError("pipeline has no bounded expanded region residency");
  SmallVector<Value> initial(schedule.loop.getInitArgs());
  auto outside = surroundingRegisterGroups(schedule.loop, initial);
  if (!outside) return schedule.loop.emitError("pipeline has no bounded surrounding residency");
  // The surrounding values cannot be reclaimed by changing this issue order.
  // If they already consume the target budget, the resource owner still has to
  // realize any required spill; the scheduler uses the smallest positive local
  // budget to retire current operands before adding another version.
  int64_t available = std::max<int64_t>(1, kernel.getTarget().getVectorRegisters() - *outside);
  auto selected = orderPipelineFrontiers(*copy, *expanded, available);
  if (failed(selected)) return failure();
  auto after = registerPeak(expanded->guard.getThenRegion().front());
  if (!after) return schedule.loop.emitError("interleaved pipeline has no bounded expanded residency");
  // Both candidates have identical reads, arithmetic, stages and carriers. If
  // pressure is unchanged, keep maximal lookahead; there is no claimed latency
  // model that could justify a different tie-break.
  if (after->total() >= before->total()) return success();
  SmallVector<Operation *> order;
  for (Operation *operation : *selected) {
    auto originalPosition = operation->getAttrOfType<IntegerAttr>(pipelineOrder).getInt();
    order.push_back(schedule.ordered[originalPosition]);
  }
  schedule.ordered = std::move(order);
  if (failed(schedule.verify())) return failure();
  schedule.write();
  ++interleaved;
  largestReduction = std::max(largestReduction, before->total() - after->total());
  return success();
}

class ScheduleRISCVLevelsPass
    : public PassWrapper<ScheduleRISCVLevelsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleRISCVLevelsPass)
  ScheduleRISCVLevelsPass() = default;
  ScheduleRISCVLevelsPass(const ScheduleRISCVLevelsPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override { return "weft-riscv-schedule-levels"; }
  StringRef getDescription() const override {
    return "Schedule dependency-derived physical stages with bounded SSA residency";
  }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.schedule")) loops.push_back(loop);
    });
    IRRewriter b(&getContext());
    unsigned selected = 0;
    int64_t reduction = 0;
    for (auto loop : loops) {
      auto binding = loop->getAttrOfType<riscv::ScheduleAttr>("weft.riscv.schedule");
      if (binding.getPipelineDepth() == 1) continue;
      bool nested = false;
      loop.getBody()->walk([&](scf::ForOp child) {
        nested |= child != loop && child->hasAttr("weft.riscv.schedule");
      });
      if (nested) {
        loop->removeAttr("weft.riscv.schedule");
        continue;
      }
      auto candidate = schedulePipeline(loop);
      if (failed(candidate)) { signalPassFailure(); return; }
      if (!*candidate) {
        // No independent supply is a fact of this graph. Consume the optional
        // local schedule without manufacturing a producer or copying a read.
        loop->removeAttr("weft.riscv.schedule");
        continue;
      }
      // A frontier that already fills the vector file cannot leave room for
      // its consumers. Try the other dependency-defined boundary: complete
      // independent arithmetic before carrying values to the accumulator.
      auto transportGroups = [&](const PipelineSchedule &schedule) -> std::optional<int64_t> {
        int64_t count = 0;
        for (Value value : schedule.transported) {
          auto groups = residentRegisterGroups(value);
          if (!groups) return std::nullopt;
          count += *groups;
        }
        return count;
      };
      auto groups = transportGroups(**candidate);
      auto kernel = loop->getParentOfType<riscv::KernelOp>();
      if (groups && *groups >= kernel.getTarget().getVectorRegisters()) {
        auto compact = schedulePipeline(loop, PipelineFrontier::Carry);
        if (failed(compact)) { signalPassFailure(); return; }
        if (*compact) {
          auto compactGroups = transportGroups(**compact);
          if (compactGroups && *compactGroups < *groups) {
            llvm::errs() << "weft-pipeline-carry-frontier: groups=" << *groups
                         << " -> " << *compactGroups << '\n';
            candidate = std::move(compact);
          }
        }
      }
      if (failed(selectFrontiers(**candidate, b, selected, reduction))) {
        signalPassFailure();
        return;
      }
    }
    interleavedLoops += selected;
    maximumReduction = std::max<int64_t>(maximumReduction, reduction);
    if (selected)
      llvm::errs() << "weft-pipeline-frontier loops=" << selected
                   << " largest-region-register-reduction=" << reduction << '\n';
  }
private:
  Statistic interleavedLoops{this, "interleaved-loops", "Pipelines with a smaller expanded region register peak"};
  Statistic maximumReduction{this, "maximum-register-reduction", "Largest reduction in expanded region register groups"};
};

// Move an independent epilogue supply ahead of the contraction it will join.
// This is same-block scheduling: no iteration, Level or numerical boundary is
// changed. Keep the selected layouts/leaves and check their complete live set.
class ScheduleRISCVLocalSuppliesPass
    : public PassWrapper<ScheduleRISCVLocalSuppliesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleRISCVLocalSuppliesPass)
private:
  static bool collectRead(Value value, Operation *anchor,
                          llvm::SetVector<Operation *> &operations) {
    auto *op = value.getDefiningOp();
    if (!op || op->getBlock() != anchor->getBlock() || op->isBeforeInBlock(anchor))
      return true;
    if (operations.contains(op)) return true;
    if (operations.size() >= 24 || op == anchor || op->getNumRegions() ||
        op->hasAttr("birth_id") ||
        !(isa<riscv::IMEFragmentStorageLoadOp, riscv::PhysicalPointOp,
              riscv::FieldOp, riscv::ExtractOp, riscv::CastOp, riscv::WidenOp>(op) ||
          op->getName().getDialectNamespace() == "arith") ||
        !riscv::hasOnlyReadEffects(op))
      return false;
    for (Value operand : op->getOperands())
      if (!collectRead(operand, anchor, operations)) return false;
    operations.insert(op);
    return true;
  }

  static unsigned scheduleReads(ModuleOp module) {
    SmallVector<riscv::IMEFragmentStorageLoadOp> reads;
    module.walk([&](riscv::IMEFragmentStorageLoadOp op) { reads.push_back(op); });
    unsigned selected = 0;
    for (auto read : reads) {
      auto kernel = read->getParentOfType<riscv::KernelOp>();
      if (!kernel || kernel.getResourcesMaterialized() ||
          riscv::fragmentPartCount(read.getResult().getType()) < 2)
        continue;
      riscv::IMEFragmentMMAOp anchor;
      for (Operation *op = read->getPrevNode(); op; op = op->getPrevNode()) {
        if (auto mma = dyn_cast<riscv::IMEFragmentMMAOp>(op)) {
          if (!mma.getMemoryClobber()) anchor = mma;
          break;
        }
        if (!riscv::hasOnlyReadEffects(op)) break;
      }
      if (!anchor) continue;
      llvm::SetVector<Operation *> closure;
      if (!collectRead(read.getResult(), anchor, closure)) continue;
      SmallVector<std::pair<Operation *, Operation *>> positions;
      for (Operation *op : closure) {
        positions.emplace_back(op, op->getNextNode());
        op->moveBefore(anchor);
      }
      selectFragmentRegisterReuse(kernel);
      auto pressure = physicalRegisterPressure(kernel);
      if (!pressure || pressure->total() > kernel.getTarget().getVectorRegisters()) {
        for (auto [op, next] : llvm::reverse(positions)) op->moveBefore(next);
        continue;
      }
      ++selected;
    }
    return selected;
  }

  static riscv::IMEUnpackOp fragmentResult(Value value) {
    SmallVector<Value> pending{value};
    llvm::SmallPtrSet<Operation *, 16> visited;
    riscv::IMEUnpackOp result;
    while (!pending.empty()) {
      auto *op = pending.pop_back_val().getDefiningOp();
      if (!op || !visited.insert(op).second) continue;
      if (visited.size() > 24) return {};
      if (auto unpack = dyn_cast<riscv::IMEUnpackOp>(op)) {
        if (result && result != unpack) return {};
        result = unpack;
        continue;
      }
      if (op->hasAttr("birth_id")) continue;
      if (isa<riscv::WidenOp, riscv::CastOp, riscv::ReshapeOp,
              riscv::ConvertLayoutOp>(op)) {
        pending.push_back(op->getOperand(0));
      } else if (auto binary = dyn_cast<riscv::BinaryOp>(op);
                 binary && (binary.getKind() == "add" || binary.getKind() == "sub" ||
                            binary.getKind() == "mul")) {
        // An independent correction does not sever the dependency on the
        // contraction. Follow only lane-local arithmetic to its unique unpack;
        // do not move or reassociate that arithmetic while scheduling supply.
        llvm::append_range(pending, binary->getOperands());
      }
    }
    return result;
  }

  // A fragment can leave an inner issue loop before its scalar/vector epilogue.
  // Follow that result's yield/carry, not merely any MMA found inside the loop.
  // The scheduling anchor stays in the consumer's block: supplies are neither
  // cloned into the loop nor moved across their enclosing Level.
  static Operation *accumulationAnchor(Value value, unsigned depth = 0) {
    if (depth >= 8) return nullptr;
    if (auto mma = value.getDefiningOp<riscv::IMEFragmentMMAOp>()) {
      Operation *first = mma;
      while (auto prior = mma.getAccumulator().getDefiningOp<riscv::IMEFragmentMMAOp>()) {
        if (prior->getBlock() != first->getBlock()) break;
        first = prior;
        mma = prior;
      }
      return first;
    }
    auto result = dyn_cast<OpResult>(value);
    auto loop = result ? dyn_cast<scf::ForOp>(result.getOwner()) : scf::ForOp();
    if (!loop) return nullptr;
    Value carried = loop.getBody()->getTerminator()->getOperand(result.getResultNumber());
    auto *nested = accumulationAnchor(carried, depth + 1);
    return nested && nested->getBlock() == loop.getBody() ? loop.getOperation() : nullptr;
  }

  static bool canCross(Operation *operation) {
    if (auto mma = dyn_cast<riscv::IMEFragmentMMAOp>(operation))
      // Its write effect belongs to the SSA fragment, not the source storage.
      return !mma.getMemoryClobber();
    if (isa<scf::ForOp, scf::IfOp>(operation)) {
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (Operation &nested : block)
            if (!canCross(&nested)) return false;
      return true;
    }
    return riscv::hasOnlyReadEffects(operation);
  }

  static bool collect(Value value, Operation *anchor, llvm::SetVector<Operation *> &operations) {
    auto *op = value.getDefiningOp();
    if (!op || op->getBlock() != anchor->getBlock() || op->isBeforeInBlock(anchor)) return true;
    if (operations.contains(op)) return true;
    if (operations.size() >= 24 || op == anchor || op->getNumRegions() ||
        op->hasAttr("birth_id") ||
        !isa<riscv::FieldOp, riscv::FieldReadOp, riscv::WidenOp, riscv::CastOp,
             riscv::BinaryOp, riscv::ReshapeOp, riscv::ConvertLayoutOp>(op) ||
        !riscv::hasOnlyReadEffects(op)) return false;
    if (auto binary = dyn_cast<riscv::BinaryOp>(op); binary && binary.getKind() != "mul") return false;
    for (Value operand : op->getOperands())
      if (!collect(operand, anchor, operations)) return false;
    operations.insert(op);
    return true;
  }

public:
  StringRef getArgument() const override { return "weft-riscv-schedule-local-supplies"; }
  StringRef getDescription() const override {
    return "Overlap independent same-block numeric supplies with selected contractions";
  }
  void runOnOperation() override {
    SmallVector<riscv::BinaryOp> consumers;
    getOperation().walk([&](riscv::IMEZeroOp zero) {
      auto kernel = zero->getParentOfType<riscv::KernelOp>();
      if (!kernel || kernel.getResourcesMaterialized() ||
          !zero->getParentOfType<scf::ForOp>() || !zero.getResult().hasOneUse())
        return;
      auto width = riscv::fragmentZeroPrefixWidth(zero.getResult().getType(), kernel.getTarget());
      if (!width) return;
      // Keep a short logical zero at its use instead of hoisting a full
      // fragment constant and copying every unused hardware row each iteration.
      OpBuilder builder(zero);
      zero.setLeafAttr(leaf(builder, "ime", "fragment-init", "ime.zero.local-prefix",
                           "ime.zero.local-prefix", 0, 0, 0, 0, "none", "exact", {*width}));
    });
    getOperation().walk([&](riscv::BinaryOp op) {
      if (op.getKind() == "mul") consumers.push_back(op);
    });
    unsigned selected = 0;
    for (auto consumer : consumers) {
      auto kernel = consumer->getParentOfType<riscv::KernelOp>();
      if (!kernel || kernel.getResourcesMaterialized()) continue;
      auto unpack = fragmentResult(consumer.getLhs());
      Value supply = consumer.getRhs();
      if (!unpack) { unpack = fragmentResult(consumer.getRhs()); supply = consumer.getLhs(); }
      if (!unpack || unpack->getBlock() != consumer->getBlock()) continue;
      auto *first = accumulationAnchor(unpack.getInput());
      if (!first || first->getBlock() != consumer->getBlock()) continue;
      llvm::SetVector<Operation *> closure;
      if (!collect(supply, first, closure) || closure.empty()) continue;
      bool legal = true;
      for (Operation *op = first; op != consumer; op = op->getNextNode())
        legal &= canCross(op);
      if (!legal) continue;
      SmallVector<std::pair<Operation *, Operation *>> positions;
      for (Operation *op : closure) {
        positions.emplace_back(op, op->getNextNode());
        op->moveBefore(first);
      }
      selectFragmentRegisterReuse(kernel);
      auto pressure = physicalRegisterPressure(kernel);
      if (!pressure || pressure->total() > kernel.getTarget().getVectorRegisters()) {
        for (auto [op, next] : llvm::reverse(positions)) op->moveBefore(next);
        continue;
      }
      ++selected;
    }
    const auto reads = scheduleReads(getOperation());
    if (selected || reads)
      llvm::errs() << "weft-local-supply-schedule: selected=" << selected
                   << " reads=" << reads << '\n';
  }
};
} // namespace

std::unique_ptr<Pass> weft::createScheduleRISCVLevelsPass() {
  return std::make_unique<ScheduleRISCVLevelsPass>();
}

std::unique_ptr<Pass> weft::createScheduleRISCVLocalSuppliesPass() {
  return std::make_unique<ScheduleRISCVLocalSuppliesPass>();
}
