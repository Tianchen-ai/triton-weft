#include "Weft/Target/X86Compiler.h"
#include "Resources.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;
namespace {
// Only closed reductions emitted by the array lowering have this marker.
// User control/carries and arbitrary associative-looking expressions are not candidates.
SmallVector<Operation *> combiners(scf::ForOp loop) {
  if (!loop->hasAttr("weft.x86.array_reduction") || loop.getNumResults() == 0 || loop.getNumResults() > 2 ||
      !isa<VectorType>(loop.getResult(0).getType())) return {};
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<Operation *> result;
  for (auto [i, value] : llvm::enumerate(yield.getOperands())) {
    auto *op = value.getDefiningOp();
    if (!op || op->getBlock() != loop.getBody() ||
        (op->getNumOperands() != 2 && !isa<math::FmaOp>(op)) ||
        !loop.getRegionIterArgs()[i].hasOneUse() ||
        !llvm::is_contained(op->getOperands(), loop.getRegionIterArgs()[i])) return {};
    if (!isa<arith::AddFOp, arith::AddIOp, arith::MaxNumFOp, arith::MinNumFOp,
             arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp, arith::OrIOp, math::FmaOp>(op)) return {};
    if (isa<math::FmaOp>(op) && op->getOperand(2) != loop.getRegionIterArgs()[i]) return {};
    result.push_back(op);
  }
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (op.getNumRegions()) return {};
    if (isa<memref::LoadOp, vector::LoadOp, vector::GatherOp>(op)) continue;
    if (!isMemoryEffectFree(&op)) return {};
  }
  // Each extra partial starts from the primitive's identity, never an author seed.
  for (Value value : loop.getInitArgs())
    if (!matchPattern(value, m_Constant())) return {};
  return result;
}
Value combine(IRRewriter &b, Operation *op, Value left, Value right) {
  if (isa<math::FmaOp>(op)) return arith::AddFOp::create(b, op->getLoc(), left, right);
  OperationState state(op->getLoc(), op->getName());
  state.addAttributes(op->getAttrs()); state.addOperands({left, right}); state.addTypes(left.getType());
  return b.create(state)->getResult(0);
}
struct PartialSchedule {
  scf::ForOp loop;
  SmallVector<Operation *> setup;
};
PartialSchedule interleave(IRRewriter &b, scf::ForOp source, unsigned copies, int64_t step) {
  b.setInsertionPoint(source); Location loc = source.getLoc();
  Operation *before = source->getPrevNode();
  Value span = arith::ConstantIndexOp::create(b, loc, step * copies);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value factor = arith::ConstantIndexOp::create(b, loc, copies);
  // Count iterations, not complete coordinate spans. A final non-aligned
  // iteration belongs to the reduction too. Clamp empty/reversed domains before
  // unsigned division; subtraction can span the full signed index range.
  Value upper = b.createOrFold<arith::MaxSIOp>(loc, source.getUpperBound(), source.getLowerBound());
  Value distance = b.createOrFold<arith::SubIOp>(loc, upper, source.getLowerBound());
  Value quotient = b.createOrFold<arith::DivUIOp>(loc, distance, source.getStep());
  Value residue = b.createOrFold<arith::RemUIOp>(loc, distance, source.getStep());
  Value extra = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, residue, zero);
  Value extraIndex = b.createOrFold<arith::IndexCastUIOp>(loc, b.getIndexType(), extra);
  Value count = b.createOrFold<arith::AddIOp>(loc, quotient, extraIndex);
  Value tail = b.createOrFold<arith::RemUIOp>(loc, count, factor);
  Value complete = b.createOrFold<arith::SubIOp>(loc, count, tail);
  Value extent = b.createOrFold<arith::MulIOp>(loc, complete, source.getStep());
  Value boundary = b.createOrFold<arith::AddIOp>(loc, source.getLowerBound(), extent);
  Value exact = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tail, zero);
  // With no tail the original upper bound already closes the last group. Do
  // not form a new observable endpoint beyond it, possibly outside index range.
  Value end = b.createOrFold<arith::SelectOp>(loc, exact, upper, boundary);
  SmallVector<Value> initial;
  for (unsigned i = 0; i < copies; ++i) llvm::append_range(initial, source.getInitArgs());
  auto loop = scf::ForOp::create(b, loc, source.getLowerBound(), end, span, initial);
  SmallVector<Operation *> setup;
  for (Operation *op = before ? before->getNextNode() : &source->getBlock()->front();
       op != loop.getOperation(); op = op->getNextNode()) setup.push_back(op);
  b.setInsertionPointToStart(loop.getBody());
  SmallVector<Value> yielded;
  for (unsigned i = 0; i < copies; ++i) {
    IRMapping mapping;
    Value index = loop.getInductionVar();
    if (i) index = arith::AddIOp::create(b, loc, index, arith::ConstantIndexOp::create(b, loc, i * step));
    mapping.map(source.getInductionVar(), index);
    mapping.map(source.getRegionIterArgs(), loop.getRegionIterArgs().slice(i * source.getNumResults(), source.getNumResults()));
    for (Operation &op : source.getBody()->without_terminator()) b.clone(op, mapping);
    for (Value value : source.getBody()->getTerminator()->getOperands()) yielded.push_back(mapping.lookupOrDefault(value));
  }
  scf::YieldOp::create(b, loc, yielded);
  return {loop, std::move(setup)};
}
class ScheduleReductions : public PassWrapper<ScheduleReductions, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleReductions)
  StringRef getArgument() const override { return "weft-x86-schedule-array-reductions"; }
  StringRef getDescription() const override {
    return "Select independent partial schedules by work and surrounding physical SSA residency";
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.x86.array_reduction")) loops.push_back(loop);
    });
    for (auto source : loops) {
      auto updates = combiners(source);
      if (updates.empty()) continue;
      APInt increment;
      if (!source.getInductionVar().getType().isIndex() || source.getUnsignedCmp() ||
          !matchPattern(source.getStep(), m_ConstantInt(&increment)) ||
          increment.getBitWidth() > 64 || !increment.isStrictlyPositive()) continue;
      int64_t step = increment.getSExtValue();
      auto model = weft::x86::ResourceModel::get(source);
      auto original = weft::x86::analyzeLoop(source, model);
      if (original.unpricedOperations) continue;
      auto tripCount = weft::analysis::staticTripCount(source);
      bool staticDomain = tripCount.has_value();
      uint64_t trips = tripCount.value_or(0);
      double mergeIssues = 0, mergeLatency = 0;
      bool mergePriced = true;
      for (auto [position, update] : llvm::enumerate(updates)) {
        b.setInsertionPoint(source);
        Value identity = source.getInitArgs()[position];
        Value merged = combine(b, update, identity, identity);
        auto estimate = weft::x86::instructionCost(merged.getDefiningOp(), model);
        if (estimate) {
          mergeIssues += estimate->issues;
          mergeLatency = std::max(mergeLatency, estimate->latency);
        } else mergePriced = false;
        b.eraseOp(merged.getDefiningOp());
      }
      if (!mergePriced) continue;
      auto finishCost = [&](unsigned copies) {
        return std::max(double(copies - 1) * mergeIssues,
                        double(llvm::Log2_64_Ceil(copies)) * mergeLatency);
      };
      // Price the actual candidate, including its expanded address/conversion
      // work and SSA recurrence. Independent chains reduce dependency pressure
      // but still pay all supply work and the final combine within the primitive.
      auto cost = [&](unsigned copies, const weft::x86::LoopResources &resources, double setup = 0) {
        auto execution = weft::x86::loopExecutionCost(resources);
        if (!staticDomain) return execution.period / copies;
        uint64_t full = trips / copies, rest = trips % copies;
        return setup + execution.total(full) + weft::x86::loopExecutionCost(original).total(rest) + finishCost(copies);
      };
      unsigned selected = 1, visited = 1, rejected = 0;
      double best = cost(1, original);
      scf::ForOp chosen;
      weft::x86::LoopResources selectedResources = original;
      SmallVector<Attribute> candidates;
      auto recordCandidate = [&](unsigned copies, const weft::x86::LoopResources &resources, StringRef decision, double setup = 0) {
        candidates.push_back(b.getDictionaryAttr({
          b.getNamedAttr("partial_copies", b.getI64IntegerAttr(copies)),
          b.getNamedAttr("estimated_cost", b.getF64FloatAttr(cost(copies, resources, setup))),
          b.getNamedAttr("setup_cost", b.getF64FloatAttr(setup)),
          b.getNamedAttr("final_combine_cost", b.getF64FloatAttr(finishCost(copies))),
          b.getNamedAttr("resources", weft::x86::resourceAttributes(b, resources)),
          b.getNamedAttr("decision", b.getStringAttr(decision))}));
      };
      recordCandidate(1, original, model.fits(original.peak) ? "legal" : "original-requires-llvm-allocation");
      // This is a finite physical policy domain, bounded by architectural capacity.
      // Every candidate is built and checked; over-budget schedules do not win.
      unsigned limit = std::min(8u, model.vectorRegisters / unsigned(source.getNumResults()));
      for (unsigned copies = 2; copies <= limit; ++copies) {
        if (staticDomain && trips < copies) break;
        if (uint64_t(step) > uint64_t(std::numeric_limits<int64_t>::max()) / copies) break;
        ++visited;
        auto schedule = interleave(b, source, copies, step);
        auto candidate = schedule.loop;
        auto resources = weft::x86::analyzeLoop(candidate, model);
        double setup = 0;
        for (Operation *operation : schedule.setup) {
          if (auto estimate = weft::x86::instructionCost(operation, model)) setup += estimate->issues;
          else ++resources.unpricedOperations;
        }
        if (resources.unpricedOperations || !model.fits(resources.peak)) {
          recordCandidate(copies, resources, resources.unpricedOperations ? "unpriced-operation" : "resource-limit", setup);
          ++rejected; b.eraseOp(candidate); continue;
        }
        recordCandidate(copies, resources, "legal", setup);
        double work = cost(copies, resources, setup);
        if (work < best) {
          if (chosen) b.eraseOp(chosen);
          chosen = candidate; best = work; selected = copies; selectedResources = resources;
        } else b.eraseOp(candidate);
      }
      NamedAttrList record(weft::x86::resourceAttributes(b, selectedResources));
      record.set("partial_copies", b.getI64IntegerAttr(selected));
      record.set("partial_vectors", b.getI64IntegerAttr(selected * source.getNumResults()));
      record.set("candidate_count", b.getI64IntegerAttr(visited));
      record.set("rejected_resource_candidates", b.getI64IntegerAttr(rejected));
      record.set("cost_basis", b.getStringAttr(staticDomain ? "candidate-supply-dependency-tail-and-merge" : "steady-state-per-packet; merge-recorded-separately"));
      record.set("candidates", b.getArrayAttr(candidates));
      record.set("estimated_selected_cost", b.getF64FloatAttr(best));
      weft::x86::updateArraySchedule(source, b, record.getDictionary(&getContext()).getValue());
      if (!chosen) continue;
      chosen->setAttr("weft.x86.partial_copies", b.getI64IntegerAttr(selected));
      b.setInsertionPointAfter(chosen);
      SmallVector<Value> partials(chosen.getResults());
      // Pairwise combining stays inside the same closed primitive and preserves
      // exact integer width, min/max policy and validity independently.
      while (partials.size() > updates.size()) {
        SmallVector<Value> next;
        for (size_t start = 0; start < partials.size(); start += 2 * updates.size())
          for (size_t i = 0; i < updates.size(); ++i)
            next.push_back(start + updates.size() < partials.size() ?
                combine(b, updates[i], partials[start + i], partials[start + updates.size() + i]) : partials[start + i]);
        partials = std::move(next);
      }
      IRMapping mapping;
      auto remainder = cast<scf::ForOp>(b.clone(*source, mapping));
      remainder.setLowerBound(chosen.getUpperBound());
      remainder.getInitArgsMutable().assign(partials);
      remainder->removeAttr("weft.x86.array_reduction");
      b.replaceOp(source, remainder.getResults());
    }
    getOperation().walk([&](scf::ForOp loop) {
      for (StringRef name : {"weft.x86.array_pointwise", "weft.x86.array_reduction", "weft.x86.array_schedule_id", "weft.x86.array_scope"})
        loop->removeAttr(name);
    });
  }
};
// Closed reductions permit a tree, but LLVM may otherwise serialize every lane
// (including each scalar NaN fixup). Materialize the tree in physical IR; no global
// fast-math flag, NaN assumption or source control reassociation is necessary.
bool excludesNegativeZero(Value value, unsigned depth = 0) {
  if (depth > 32) return false;
  if (value.getDefiningOp<math::AbsFOp>()) return true;
  Attribute constant;
  if (matchPattern(value, m_Constant(&constant))) {
    auto safe = [](const APFloat &v) { return !(v.isZero() && v.isNegative()); };
    if (auto scalar = dyn_cast<FloatAttr>(constant)) return safe(scalar.getValue());
    if (auto dense = dyn_cast<DenseFPElementsAttr>(constant))
      return llvm::all_of(dense.getValues<APFloat>(), safe);
  }
  auto recurse = [&](Value operand) { return excludesNegativeZero(operand, depth + 1); };
  if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
    int64_t width = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
    bool left = false, right = false;
    for (int64_t lane : shuffle.getMask()) { left |= lane < width; right |= lane >= width; }
    return (!left || recurse(shuffle.getV1())) && (!right || recurse(shuffle.getV2()));
  }
  if (auto select = value.getDefiningOp<arith::SelectOp>())
    return recurse(select.getTrueValue()) && recurse(select.getFalseValue());
  auto *op = value.getDefiningOp();
  if (isa_and_nonnull<vector::BroadcastOp, vector::ExtractOp, vector::ExtractStridedSliceOp,
                      vector::ShapeCastOp, arith::ExtFOp>(op)) return recurse(op->getOperand(0));
  return false;
}

bool excludesNegativeInfinity(Value value, unsigned depth = 0) {
  if (depth > 32) return false;
  if (value.getDefiningOp<math::AbsFOp>()) return true;
  Attribute constant;
  if (matchPattern(value, m_Constant(&constant))) {
    auto safe = [](const APFloat &v) { return !(v.isInfinity() && v.isNegative()); };
    if (auto scalar = dyn_cast<FloatAttr>(constant)) return safe(scalar.getValue());
    if (auto dense = dyn_cast<DenseFPElementsAttr>(constant))
      return llvm::all_of(dense.getValues<APFloat>(), safe);
  }
  auto recurse = [&](Value input) { return excludesNegativeInfinity(input, depth + 1); };
  if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
    int64_t width = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
    bool left = false, right = false;
    for (int64_t lane : shuffle.getMask()) { left |= lane < width; right |= lane >= width; }
    return (!left || recurse(shuffle.getV1())) && (!right || recurse(shuffle.getV2()));
  }
  if (auto select = value.getDefiningOp<arith::SelectOp>())
    return recurse(select.getTrueValue()) && recurse(select.getFalseValue());
  auto *op = value.getDefiningOp();
  if (isa_and_nonnull<vector::BroadcastOp, vector::ExtractOp, vector::ExtractStridedSliceOp,
                     vector::ShapeCastOp, arith::ExtFOp>(op)) return recurse(op->getOperand(0));
  return false;
}

void combineAbsoluteMaxima(ModuleOp module, IRRewriter &b) {
  SmallVector<vector::FromElementsOp> packs;
  module.walk([&](vector::FromElementsOp pack) { packs.push_back(pack); });
  for (auto pack : packs) {
    auto output = cast<VectorType>(pack.getType());
    if (output.isScalable() || output.getRank() != 1 || !output.getElementType().isF32() ||
        output.getNumElements() < 4 || output.getNumElements() > 16 || output.getNumElements() % 4) continue;
    SmallVector<vector::ReductionOp> reductions;
    bool legal = true;
    for (Value result : pack.getOperands()) {
      auto reduction = result.getDefiningOp<vector::ReductionOp>();
      if (!reduction || !result.hasOneUse() || reduction->getBlock() != pack->getBlock() ||
          reduction.getKind() != vector::CombiningKind::MAXNUMF || reduction.getAcc()) { legal = false; break; }
      auto input = cast<VectorType>(reduction.getVector().getType());
      if (input.isScalable() || input.getRank() != 1 || input.getNumElements() < 4 ||
          input.getNumElements() > 32 || !llvm::isPowerOf2_64(input.getNumElements()) ||
          !excludesNegativeZero(reduction.getVector()) || !excludesNegativeInfinity(reduction.getVector())) {
        legal = false; break;
      }
      reductions.push_back(reduction);
    }
    if (!legal) continue;
    Location loc = pack.getLoc();
    auto flags = arith::FastMathFlags::nnan | arith::FastMathFlags::nsz;
    auto maximum = [&](Value a, Value c) -> Value { return arith::MaxNumFOp::create(b, loc, a, c, flags); };
    auto shuffle = [&](Value a, Value c, ArrayRef<int64_t> indices) -> Value {
      return vector::ShuffleOp::create(b, loc, a, c, indices);
    };
    SmallVector<Value> partials;
    for (auto reduction : reductions) {
      b.setInsertionPoint(reduction);
      Value value = reduction.getVector();
      Value numeric = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ORD, value, value);
      Value identity = arith::ConstantOp::create(b, loc, value.getType(),
          DenseElementsAttr::get(cast<VectorType>(value.getType()), b.getF32FloatAttr(-std::numeric_limits<float>::infinity())));
      value = arith::SelectOp::create(b, loc, numeric, value, identity);
      // Retain four lanes beside each original reduction. Only this final
      // native lane cohort stays live until the common output assembly.
      for (int64_t count = cast<VectorType>(value.getType()).getNumElements(); count > 4; count /= 2) {
        SmallVector<int64_t> low, high;
        for (int64_t i = 0; i < count / 2; ++i) { low.push_back(i); high.push_back(i + count / 2); }
        value = maximum(shuffle(value, value, low), shuffle(value, value, high));
      }
      partials.push_back(value);
    }
    b.setInsertionPoint(pack);
    Value outputValue = arith::ConstantOp::create(b, loc, output, b.getZeroAttr(output));
    for (int64_t first = 0; first < output.getNumElements(); first += 4) {
      auto pair = [&](Value a, Value c) {
        return maximum(shuffle(a, c, {0, 1, 4, 5}), shuffle(a, c, {2, 3, 6, 7}));
      };
      Value a = pair(partials[first], partials[first + 1]);
      Value c = pair(partials[first + 2], partials[first + 3]);
      Value value = maximum(shuffle(a, c, {0, 2, 4, 6}), shuffle(a, c, {1, 3, 5, 7}));
      auto packet = cast<VectorType>(value.getType());
      Value identity = arith::ConstantOp::create(b, loc, packet,
          DenseElementsAttr::get(packet, b.getF32FloatAttr(-std::numeric_limits<float>::infinity())));
      Value nan = arith::ConstantOp::create(b, loc, packet,
          DenseElementsAttr::get(packet, b.getF32FloatAttr(std::numeric_limits<float>::quiet_NaN())));
      Value numeric = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ONE, value, identity);
      value = arith::SelectOp::create(b, loc, numeric, value, nan);
      outputValue = vector::InsertStridedSliceOp::create(b, loc, value, outputValue,
          ArrayRef<int64_t>{first}, ArrayRef<int64_t>{1});
    }
    b.replaceOp(pack, outputValue);
    for (auto reduction : reductions) b.eraseOp(reduction);
  }
}

class ReductionTrees : public PassWrapper<ReductionTrees, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReductionTrees)
  StringRef getArgument() const override { return "weft-x86-materialize-reduction-trees"; }
  StringRef getDescription() const override { return "Materialize balanced trees within closed floating SIMD reductions"; }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    combineAbsoluteMaxima(getOperation(), b);
    SmallVector<vector::ReductionOp> reductions;
    getOperation().walk([&](vector::ReductionOp op) {
      auto type = cast<VectorType>(op.getVector().getType());
      auto kind = op.getKind();
      bool supported = kind == vector::CombiningKind::ADD || kind == vector::CombiningKind::MAXNUMF ||
          kind == vector::CombiningKind::MINNUMF || kind == vector::CombiningKind::MAXIMUMF ||
          kind == vector::CombiningKind::MINIMUMF;
      if (supported && isa<FloatType>(type.getElementType()) &&
          type.getRank() == 1 && !type.isScalable() && !op.getAcc()) reductions.push_back(op);
    });
    for (auto reduction : reductions) {
      b.setInsertionPoint(reduction);
      Value value = reduction.getVector();
      int64_t count = cast<VectorType>(value.getType()).getNumElements();
      auto kind = reduction.getKind();
      bool numberExtremum = count >= 4 && (kind == vector::CombiningKind::MAXNUMF ||
                                          kind == vector::CombiningKind::MINNUMF);
      bool noNegativeZero = numberExtremum && excludesNegativeZero(value);
      bool distinctIdentity = numberExtremum && kind == vector::CombiningKind::MAXNUMF &&
          excludesNegativeInfinity(value);
      Value anyNumber;
      if (numberExtremum) {
        // Resolve NaNs once for the whole closed primitive. Every following
        // tree input is numeric, so each combine needs no repeated NaN fixup.
        auto type = cast<VectorType>(value.getType());
        Value numeric = arith::CmpFOp::create(b, reduction.getLoc(), arith::CmpFPredicate::ORD, value, value);
        if (!distinctIdentity)
          anyNumber = vector::ReductionOp::create(b, reduction.getLoc(), vector::CombiningKind::OR, numeric);
        double identity = kind == vector::CombiningKind::MAXNUMF ?
            -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        Value neutral = arith::ConstantOp::create(b, reduction.getLoc(), type,
            DenseElementsAttr::get(type, b.getFloatAttr(type.getElementType(), identity)));
        value = arith::SelectOp::create(b, reduction.getLoc(), numeric, value, neutral);
      }
      while (count > 1) {
        SmallVector<int64_t> left, right;
        if (reduction.getKind() == vector::CombiningKind::ADD) {
          // Keep each shrinking packet contiguous. Interleaved even/odd
          // shuffles cross x86 register and 128-bit lane boundaries at every
          // level; folding halves needs one extraction and a packed add.
          // The closed sum owns this tree. No source carry, cast or scale is
          // moved, and an unpaired lane still combines with positive zero.
          int64_t half = (count + 1) / 2;
          for (int64_t i = 0; i < half; ++i) {
            left.push_back(i);
            right.push_back(i + half < count ? i + half : count);
          }
        } else {
          // Fold contiguous halves so native packets can combine directly,
          // instead of interleaving even/odd lanes across register boundaries.
          int64_t half = (count + 1) / 2;
          for (int64_t i = 0; i < half; ++i) {
            left.push_back(i);
            // An unpaired extremum combines with itself, preserving negative
            // maxima, positive minima and all-NaN input without zero padding.
            right.push_back(i + half < count ? i + half : i);
          }
        }
        Value zeros = arith::ConstantOp::create(b, reduction.getLoc(), value.getType(), b.getZeroAttr(value.getType()));
        Value a = vector::ShuffleOp::create(b, reduction.getLoc(), value, zeros, left);
        Value c = vector::ShuffleOp::create(b, reduction.getLoc(), value, zeros, right);
        switch (reduction.getKind()) {
        case vector::CombiningKind::ADD: value = arith::AddFOp::create(b, reduction.getLoc(), a, c); break;
        case vector::CombiningKind::MAXNUMF:
        case vector::CombiningKind::MINNUMF: {
          bool maximum = kind == vector::CombiningKind::MAXNUMF;
          if (!numberExtremum) {
            value = maximum ? Value(arith::MaxNumFOp::create(b, reduction.getLoc(), a, c)) :
                              Value(arith::MinNumFOp::create(b, reduction.getLoc(), a, c));
          } else {
            // Signed zero cannot change which nonzero number wins. Repair the
            // final zero from the original snapshot once, rather than carrying
            // IEEE zero selection through every internal tree node.
            auto flags = arith::FastMathFlags::nnan | arith::FastMathFlags::nsz;
            value = maximum ? Value(arith::MaxNumFOp::create(b, reduction.getLoc(), a, c, flags)) :
                              Value(arith::MinNumFOp::create(b, reduction.getLoc(), a, c, flags));
          }
          break;
        }
        case vector::CombiningKind::MAXIMUMF: value = arith::MaximumFOp::create(b, reduction.getLoc(), a, c); break;
        case vector::CombiningKind::MINIMUMF: value = arith::MinimumFOp::create(b, reduction.getLoc(), a, c); break;
        default: llvm_unreachable("unsupported reduction tree");
        }
        count = (count + 1) / 2;
      }
      Value result = vector::ExtractOp::create(b, reduction.getLoc(), value, ArrayRef<int64_t>{0});
      if (numberExtremum) {
        if (!noNegativeZero) {
          Location loc = reduction.getLoc();
          Type element = result.getType();
          Value zero = arith::ConstantOp::create(b, loc, element, b.getFloatAttr(element, 0.0));
          Value isZero = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, result, zero);
          auto correction = scf::IfOp::create(b, loc, TypeRange{element}, isZero, true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&correction.getThenRegion().front());
            auto sourceType = cast<VectorType>(reduction.getVector().getType());
            unsigned width = element.getIntOrFloatBitWidth();
            auto bitsType = sourceType.clone(b.getIntegerType(width));
            Value bits = arith::BitcastOp::create(b, loc, bitsType, reduction.getVector());
            bool maximum = kind == vector::CombiningKind::MAXNUMF;
            APInt preferredBits = maximum ? APInt(width, 0) : APInt::getSignMask(width);
            Value preferred = arith::ConstantOp::create(b, loc, bitsType,
                DenseIntElementsAttr::get(bitsType, preferredBits));
            Value matches = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bits, preferred);
            Value any = vector::ReductionOp::create(b, loc, vector::CombiningKind::OR, matches);
            Value negative = arith::ConstantOp::create(b, loc, element, b.getFloatAttr(element, -0.0));
            Value fixed = arith::SelectOp::create(b, loc, any,
                maximum ? zero : negative, maximum ? negative : zero);
            scf::YieldOp::create(b, loc, fixed);
            b.setInsertionPointToStart(&correction.getElseRegion().front());
            scf::YieldOp::create(b, loc, result);
          }
          b.setInsertionPointAfter(correction);
          result = correction.getResult(0);
        }
        // A maximum of abs values cannot equal the sanitizing -infinity
        // unless every input was NaN. Reuse that result instead of performing
        // an independent predicate reduction; no finite-domain assumption is made.
        if (distinctIdentity) {
          Value identity = arith::ConstantOp::create(b, reduction.getLoc(), result.getType(),
              b.getFloatAttr(result.getType(), -std::numeric_limits<double>::infinity()));
          anyNumber = arith::CmpFOp::create(b, reduction.getLoc(), arith::CmpFPredicate::ONE, result, identity);
        }
        Value nan = arith::ConstantOp::create(b, reduction.getLoc(), result.getType(),
            b.getFloatAttr(result.getType(), std::numeric_limits<double>::quiet_NaN()));
        result = arith::SelectOp::create(b, reduction.getLoc(), anyNumber, result, nan);
      }
      b.replaceOp(reduction, result);
    }
    // Arith maxnum/minnum leave a mixed signed-zero tie unspecified. The
    // Canonical contract requires IEEE maximumNumber/minimumNumber, including
    // that tie for pointwise operations and complete reduction results.
    SmallVector<Operation *> extrema;
    getOperation().walk([&](Operation *op) {
      if (isa<arith::MaxNumFOp, arith::MinNumFOp>(op)) extrema.push_back(op);
    });
    for (Operation *op : extrema) {
      auto flags = op->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
      if (flags && bitEnumContainsAll(flags.getValue(), arith::FastMathFlags::nnan | arith::FastMathFlags::nsz)) continue;
      auto nonzeroConstant = [](Value value) {
        Attribute constant;
        if (!matchPattern(value, m_Constant(&constant))) return false;
        if (auto scalar = dyn_cast<FloatAttr>(constant)) return !scalar.getValue().isZero();
        if (auto dense = dyn_cast<DenseFPElementsAttr>(constant))
          return llvm::all_of(dense.getValues<APFloat>(), [](const APFloat &v) { return !v.isZero(); });
        return false;
      };
      // A mixed-zero tie is impossible with a nonzero bound, or when neither
      // input can carry negative zero. Keep the simpler number operation then.
      if (nonzeroConstant(op->getOperand(0)) || nonzeroConstant(op->getOperand(1)) ||
          (excludesNegativeZero(op->getOperand(0)) && excludesNegativeZero(op->getOperand(1)))) continue;
      b.setInsertionPoint(op);
      Value lhs = op->getOperand(0), rhs = op->getOperand(1);
      Value lhsNan = arith::CmpFOp::create(b, op->getLoc(), arith::CmpFPredicate::UNO, lhs, lhs);
      Value rhsNan = arith::CmpFOp::create(b, op->getLoc(), arith::CmpFPredicate::UNO, rhs, rhs);
      Value a = arith::SelectOp::create(b, op->getLoc(), lhsNan, rhs, lhs);
      Value c = arith::SelectOp::create(b, op->getLoc(), rhsNan, lhs, rhs);
      Value result = isa<arith::MaxNumFOp>(op) ?
          Value(arith::MaximumFOp::create(b, op->getLoc(), a, c)) :
          Value(arith::MinimumFOp::create(b, op->getLoc(), a, c));
      b.replaceOp(op, result);
    }
  }
};
} // namespace
std::unique_ptr<Pass> weft::createX86ReductionSchedulePass() { return std::make_unique<ScheduleReductions>(); }
std::unique_ptr<Pass> weft::createX86ReductionTreePass() { return std::make_unique<ReductionTrees>(); }
