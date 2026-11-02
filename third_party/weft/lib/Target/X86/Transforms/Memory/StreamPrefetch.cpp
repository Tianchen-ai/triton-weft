#include "Weft/Target/X86/X86Compiler.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <limits>

using namespace mlir;
namespace {
constexpr int64_t maxLookahead = 4;
constexpr int64_t minLookaheadWindows = 4;
constexpr int64_t cacheResidentBudget = 1 << 20;
constexpr int64_t hintSpan = 64;
constexpr int64_t minFootprint = 8 * hintSpan;
constexpr int64_t maxFootprint = 32768;
constexpr int64_t maxAddressOps = 1024;
// Charge both hint instructions and their shared address graph. A contiguous
// stream can amortize many hints over a short address chain; interleaved
// packets must also pay for rebuilding their longer address graph.
constexpr int64_t maxPrefetchWork = 128;
bool readOnlyArgument(Value pointer) {
  if (cast<LLVM::LLVMPointerType>(pointer.getType()).getAddressSpace() != 0) return false;
  while (auto gep = pointer.getDefiningOp<LLVM::GEPOp>()) pointer = gep.getBase();
  auto argument = dyn_cast<BlockArgument>(pointer);
  if (!argument) return false;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front() &&
         bool(function.getArgAttr(argument.getArgNumber(), "llvm.readonly"));
}

// Recompute only a pure scalar address at another existing loop iteration.
// Carried numerical state and reads cannot become address dependencies.
bool addressClosure(Value value, scf::ForOp loop,
                    llvm::SmallPtrSetImpl<Operation *> &seen,
                    SmallVectorImpl<Operation *> &ordered, bool &varying) {
  if (value == loop.getInductionVar()) { varying = true; return true; }
  if (loop.isDefinedOutsideOfLoop(value)) return true;
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != loop.getBody() || op->getNumRegions() ||
      !isMemoryEffectFree(op) || !isSpeculatable(op) ||
      !llvm::all_of(op->getResultTypes(), [](Type type) {
        return type.isIntOrIndex() || isa<LLVM::LLVMPointerType>(type);
      })) return false;
  if (seen.contains(op)) return true;
  if (seen.size() == maxAddressOps) return false;
  seen.insert(op);
  for (Value operand : op->getOperands())
    if (!addressClosure(operand, loop, seen, ordered, varying)) return false;
  ordered.push_back(op);
  return true;
}

class StreamPrefetch : public PassWrapper<StreamPrefetch, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StreamPrefetch)
  StringRef getArgument() const override { return "weft-x86-prefetch-streams"; }
  StringRef getDescription() const override { return "Prefetch proven complete input packets at bounded future iterations"; }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) { loops.push_back(loop); });
    int64_t selected = 0;
    SmallVector<Attribute> schedules;
    for (auto loop : loops) {
      if (!loop.getInductionVar().getType().isIndex()) continue;
      APInt start, step;
      if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&start)) || start.isNegative() ||
          !matchPattern(loop.getStep(), m_ConstantInt(&step)) || step.isNegative() || step.isZero() ||
          step.getZExtValue() > uint64_t(std::numeric_limits<int64_t>::max() /
                                        (maxLookahead * minLookaheadWindows))) continue;
      bool flat = true;
      for (Operation &op : *loop.getBody()) {
        if (op.getNumRegions()) flat = false;
        if (auto call = dyn_cast<func::CallOp>(op))
          if (call.getCallee().starts_with("llvm.x86.tile") || call.getCallee().starts_with("llvm.x86.tdp")) flat = false;
      }
      if (!flat) continue;
      struct Packet { LLVM::MaskedLoadOp load; Value pointer; int64_t bytes; SmallVector<Operation *> address; };
      SmallVector<Packet> packets;
      llvm::SmallPtrSet<Operation *, 32> allAddresses;
      int64_t totalBytes = 0, hints = 0;
      for (Operation &op : *loop.getBody()) {
        auto load = dyn_cast<LLVM::MaskedLoadOp>(op);
        if (!load) continue;
        DenseElementsAttr mask;
        if (!matchPattern(load.getMask(), m_Constant(&mask)) || !mask.isSplat() ||
            !mask.getSplatValue<APInt>().isOne()) continue;
        auto type = dyn_cast<VectorType>(load.getResult().getType());
        if (!type || type.isScalable()) continue;
        int64_t bytes = type.getNumElements() * type.getElementTypeBitWidth() / 8;
        Value pointer = load.getOperand(0);
        if (bytes < 32 || bytes > maxFootprint || !readOnlyArgument(pointer)) continue;
        llvm::SmallPtrSet<Operation *, 32> seen;
        SmallVector<Operation *> address;
        bool varying = false;
        if (!addressClosure(pointer, loop, seen, address, varying) || !varying) continue;
        for (Operation *op : address) allAddresses.insert(op);
        packets.push_back({load, pointer, bytes, std::move(address)});
        totalBytes += bytes;
        hints += (bytes + hintSpan - 1) / hintSpan;
      }
      // Bound both added instructions and the outstanding hinted footprint.
      // Small record streams already fit a few cache lines. Replicating their
      // addresses and adding a guarded hint sequence can cost more than the
      // loads; leave these windows to hardware streaming. This is a physical
      // scheduling budget, independent of the operation or Encoding name.
      if (totalBytes < minFootprint || totalBytes > maxFootprint || hints > 512 ||
          allAddresses.size() > maxAddressOps ||
          allAddresses.size() + hints > maxPrefetchWork) continue;
      // Count the complete finite stream, including enclosing traversals.
      // A short traversal already contained in the lookahead budget does not
      // amortize a second address stream and guarded hints. Unknown extents
      // retain their symbolic traversal and the existing bounded lookahead.
      auto function = loop->getParentOfType<func::FuncOp>();
      auto frequency = weft::analysis::executionMultiplicityInBlock(
          packets.back().load, &function.getBody().front());
      if (frequency && frequency->dynamicLoops.empty() &&
          frequency->fixed <= uint64_t(maxFootprint / totalBytes)) continue;
      // Larger source windows retire fewer iterations while filling the same
      // bounded hint footprint. The source tile and its traversal stay intact.
      int64_t lookahead = std::min(maxLookahead, maxFootprint / totalBytes);
      OpBuilder b(packets.back().load);
      // A short counted stream pays a second address stream and guarded hints
      // for almost its whole execution. Require several complete lookahead
      // windows. For dynamic domains this remains an invocation-time decision,
      // rather than assuming that an unknown traversal is long.
      b.setInsertionPoint(loop);
      Location loc = loop.getLoc();
      Value span = arith::SubIOp::create(b, loc, loop.getUpperBound(), loop.getLowerBound());
      Value minimum = arith::ConstantIndexOp::create(b, loc,
          step.getSExtValue() * lookahead * minLookaheadWindows);
      Value longStream = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, span, minimum);
      // An inner stream can be short while a surrounding traversal streams a
      // large working set. Keep that useful prefetching: compare the complete
      // traversal against a bounded cache-resident budget without multiplying
      // dynamic trip counts (which could overflow).
      Value budget = arith::ConstantIndexOp::create(b, loc, cacheResidentBudget / totalBytes);
      Value one = arith::ConstantIndexOp::create(b, loc, 1);
      Value zero = arith::ConstantIndexOp::create(b, loc, 0);
      Value largeTraversal = arith::ConstantIntOp::create(b, loc, 0, 1);
      for (Operation *parent = loop; parent && parent != function.getOperation();
           parent = parent->getParentOp()) {
        auto counted = dyn_cast<scf::ForOp>(parent);
        if (!counted || !counted.getInductionVar().getType().isIndex()) continue;
        Value extent = arith::SubIOp::create(b, loc, counted.getUpperBound(), counted.getLowerBound());
        extent = arith::MaxSIOp::create(b, loc, extent, zero);
        Value quotient = arith::DivUIOp::create(b, loc, extent, counted.getStep());
        Value remainder = arith::RemUIOp::create(b, loc, extent, counted.getStep());
        Value tail = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, remainder, zero);
        Value increment = arith::IndexCastUIOp::create(b, loc, b.getIndexType(), tail);
        Value count = arith::AddIOp::create(b, loc, quotient, increment);
        Value exceeds = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, count, budget);
        largeTraversal = arith::OrIOp::create(b, loc, largeTraversal, exceeds);
        Value divisor = arith::MaxUIOp::create(b, loc, count, one);
        budget = arith::DivUIOp::create(b, loc, budget, divisor);
      }
      longStream = arith::OrIOp::create(b, loc, longStream, largeTraversal);
      b.setInsertionPointAfter(packets.back().load);
      Value distance = arith::ConstantIndexOp::create(b, loc, step.getSExtValue() * lookahead);
      Value remaining = arith::SubIOp::create(b, loc, loop.getUpperBound(), loop.getInductionVar());
      Value available = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, remaining, distance);
      available = arith::AndIOp::create(b, loc, available, longStream);
      auto guard = scf::IfOp::create(b, loc, available, false);
      b.setInsertionPointToStart(guard.thenBlock());
      // The subtraction and guard prove the addition cannot overflow and the
      // hinted iteration belongs to the original nonnegative counted domain.
      Value future = arith::AddIOp::create(b, loc, loop.getInductionVar(), distance);
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), future);
      auto prefetch = getOperation().lookupSymbol<func::FuncOp>("llvm.prefetch.p0");
      if (!prefetch) {
        OpBuilder::InsertionGuard insertion(b);
        b.setInsertionPointToEnd(getOperation().getBody());
        prefetch = func::FuncOp::create(b, loc, "llvm.prefetch.p0",
            b.getFunctionType({LLVM::LLVMPointerType::get(&getContext()), b.getI32Type(), b.getI32Type(), b.getI32Type()}, {}));
        prefetch.setPrivate();
      }
      Value read = arith::ConstantIntOp::create(b, loc, 0, 32);
      Value locality = arith::ConstantIntOp::create(b, loc, 3, 32);
      Value data = arith::ConstantIntOp::create(b, loc, 1, 32);
      for (const auto &packet : packets) {
        for (Operation *op : packet.address)
          if (!mapping.contains(op->getResult(0))) b.clone(*op, mapping);
        Value base = mapping.lookupOrDefault(packet.pointer);
        for (int64_t first = 0; first < packet.bytes; first += hintSpan) {
          Value last = arith::ConstantIntOp::create(b, loc, std::min(first + hintSpan, packet.bytes) - 1, 64);
          Value pointer = LLVM::GEPOp::create(b, loc, base.getType(), b.getI8Type(), base, ValueRange{last});
          func::CallOp::create(b, loc, prefetch, ValueRange{pointer, read, locality, data});
          ++selected;
        }
      }
      schedules.push_back(b.getDictionaryAttr({
          b.getNamedAttr("bytes_per_iteration", b.getI64IntegerAttr(totalBytes)),
          b.getNamedAttr("lookahead_iterations", b.getI64IntegerAttr(lookahead)),
          b.getNamedAttr("minimum_iterations", b.getI64IntegerAttr(lookahead * minLookaheadWindows + 1)),
          b.getNamedAttr("cache_resident_budget", b.getI64IntegerAttr(cacheResidentBudget)),
          b.getNamedAttr("hints_per_iteration", b.getI64IntegerAttr(hints)),
          b.getNamedAttr("address_operations", b.getI64IntegerAttr(allAddresses.size()))}));
    }
    if (selected) {
      getOperation()->setAttr("weft.x86.prefetched_streams",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), selected));
      getOperation()->setAttr("weft.x86.stream_schedules", ArrayAttr::get(&getContext(), schedules));
    }
  }
};
}
std::unique_ptr<Pass> weft::createX86StreamPrefetchPass() { return std::make_unique<StreamPrefetch>(); }
