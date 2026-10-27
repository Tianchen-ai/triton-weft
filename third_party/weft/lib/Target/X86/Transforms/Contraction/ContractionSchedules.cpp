#include "Weft/Target/X86/X86Compiler.h"
#include "Analysis/Resources.h"
#include "Transforms/Contraction/ContractionOutputs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace {
// Prove the coefficient of the reduction induction variable. Runtime bases
// remain SSA expressions. Unknown strides keep their individual gather indices.
std::optional<int64_t> coefficient(Value value, scf::ForOp loop, unsigned depth = 0) {
  if (value == loop.getInductionVar()) return 1;
  if (value.getParentBlock() != loop.getBody()) return 0;
  if (depth > 24) return {};
  Operation *op = value.getDefiningOp();
  if (!op) return {};
  if (isa<arith::ConstantOp>(op)) return 0;
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    auto left = coefficient(op->getOperand(0), loop, depth + 1);
    auto right = coefficient(op->getOperand(1), loop, depth + 1);
    if (left && right) {
      int64_t result;
      bool overflow = isa<arith::AddIOp>(op) ? llvm::AddOverflow(*left, *right, result) : llvm::SubOverflow(*left, *right, result);
      if (!overflow) return result;
    }
  }
  if (auto product = dyn_cast<arith::MulIOp>(op)) {
    APInt scale;
    for (unsigned i = 0; i < 2; ++i)
      if (matchPattern(product->getOperand(i), m_ConstantInt(&scale)) && scale.isSignedIntN(32))
        if (auto other = coefficient(product->getOperand(1-i), loop, depth + 1)) {
          int64_t result;
          if (!llvm::MulOverflow(*other, scale.getSExtValue(), result)) return result;
        }
  }
  if (auto select = dyn_cast<arith::SelectOp>(op)) {
    if (coefficient(select.getCondition(), loop, depth + 1) != std::optional<int64_t>(0)) return {};
    auto left = coefficient(select.getTrueValue(), loop, depth + 1);
    auto right = coefficient(select.getFalseValue(), loop, depth + 1);
    if (left && right && *left == *right) return left;
  }
  return {};
}

scf::ForOp packetize(IRRewriter &b, scf::ForOp source, unsigned lanes) {
  Location loc = source.getLoc(); b.setInsertionPoint(source);
  auto vectorType = [&](Type type) { return VectorType::get({int64_t(lanes)}, type); };
  auto broadcast = [&](Value value) -> Value {
    return isa<VectorType>(value.getType()) ? value : Value(vector::BroadcastOp::create(b, loc, vectorType(value.getType()), value));
  };
  Value width = arith::ConstantIndexOp::create(b, loc, lanes);
  Value end = arith::SubIOp::create(b, loc, source.getUpperBound(),
      arith::RemUIOp::create(b, loc, source.getUpperBound(), width));
  SmallVector<Value> initial;
  for (Value value : source.getInitArgs()) initial.push_back(broadcast(value));
  auto loop = scf::ForOp::create(b, loc, source.getLowerBound(), end, width, initial);
  b.setInsertionPointToStart(loop.getBody());
  IRMapping mapping, scalar;
  scalar.map(source.getInductionVar(), loop.getInductionVar());
  auto scalarIndex = [&](auto &&self, Value value) -> Value {
    if (scalar.contains(value)) return scalar.lookup(value);
    if (value.getParentBlock() != source.getBody()) return value;
    auto *op = value.getDefiningOp();
    for (Value input : op->getOperands()) scalar.map(input, self(self, input));
    return b.clone(*op, scalar)->getResult(0);
  };
  SmallVector<int64_t> offsets;
  for (unsigned i = 0; i < lanes; ++i) offsets.push_back(i);
  Value sequence = arith::ConstantOp::create(b, loc, vectorType(b.getIndexType()),
      DenseIntElementsAttr::get(vectorType(b.getIndexType()), offsets));
  Value indices = arith::AddIOp::create(b, loc, broadcast(loop.getInductionVar()), sequence);
  mapping.map(source.getInductionVar(), indices);
  mapping.map(source.getRegionIterArgs(), loop.getRegionIterArgs());
  for (Operation &op : source.getBody()->without_terminator()) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      Value index = mapping.lookupOrDefault(load.getIndices()[0]);
      Value value;
      auto stride = coefficient(load.getIndices()[0], source);
      if (stride && *stride == 1 && !load.getType().isInteger(1)) {
        Value base = scalarIndex(scalarIndex, load.getIndices()[0]);
        value = vector::LoadOp::create(b, loc, vectorType(load.getType()), load.getMemRef(), base);
      } else if (!isa<VectorType>(index.getType())) {
        value = memref::LoadOp::create(b, loc, load.getMemRef(), index);
      } else {
        Value zero = arith::ConstantIndexOp::create(b, loc, 0);
        Value all = arith::ConstantOp::create(b, loc, vectorType(b.getI1Type()),
            DenseElementsAttr::get(vectorType(b.getI1Type()), b.getBoolAttr(true)));
        Value fill = arith::ConstantOp::create(b, loc, vectorType(load.getType()), b.getZeroAttr(vectorType(load.getType())));
        value = vector::GatherOp::create(b, loc, vectorType(load.getType()), load.getMemRef(), ValueRange{zero}, index, all, fill);
      }
      mapping.map(load.getResult(), value);
    } else {
      SmallVector<Value> inputs;
      bool shaped = false;
      for (Value input : op.getOperands()) {
        Value value = mapping.lookupOrDefault(input); inputs.push_back(value);
        shaped |= isa<VectorType>(value.getType());
      }
      if (shaped) for (Value &value : inputs) value = broadcast(value);
      OperationState state(op.getLoc(), op.getName());
      state.addAttributes(op.getAttrs()); state.addOperands(inputs);
      for (Type type : op.getResultTypes()) state.addTypes(shaped ? Type(vectorType(type)) : type);
      mapping.map(op.getResults(), b.create(state)->getResults());
    }
  }
  SmallVector<Value> yields;
  for (Value value : source.getBody()->getTerminator()->getOperands())
    yields.push_back(broadcast(mapping.lookupOrDefault(value)));
  scf::YieldOp::create(b, loc, yields);
  return loop;
}

void eraseDeadPure(IRRewriter &b, scf::ForOp loop) {
  bool changed;
  do {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*loop.getBody())))
      if (!op.getNumRegions() && !op.hasTrait<OpTrait::IsTerminator>() && op.use_empty() && isMemoryEffectFree(&op)) {
        b.eraseOp(&op); changed = true;
      }
  } while (changed);
}
std::optional<double> outerWork(scf::ForOp outer, scf::ForOp reduction,
                                const weft::x86::ResourceModel &model) {
  if (!outer) return {};
  double result = 3;
  for (Operation &op : *outer.getBody()) {
    if (&op == reduction.getOperation()) continue;
    auto estimate = weft::x86::instructionCost(&op, model);
    if (!estimate) return {};
    result += estimate->issues;
  }
  return result;
}
double iterationCost(const weft::x86::LoopResources &resources) {
  return weft::x86::loopExecutionCost(resources).period;
}

class ScheduleContractions : public PassWrapper<ScheduleContractions, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleContractions)
  StringRef getArgument() const override { return "weft-x86-schedule-array-contractions"; }
  StringRef getDescription() const override {
    return "Select SIMD reduction packets for closed contractions using address and live-resource costs";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, memref::MemRefDialect, vector::VectorDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.x86.array_contraction")) loops.push_back(loop);
    });
    IRRewriter b(&getContext());
    for (auto source : loops) {
      source->removeAttr("weft.x86.array_contraction");
      if (source.getNumResults() < 1 || source.getNumResults() > 2) continue;
      APInt lower, step, count;
      if (!matchPattern(source.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
          !matchPattern(source.getStep(), m_ConstantInt(&step)) || !step.isOne()) continue;
      bool supported = true;
      unsigned bits = 8;
      for (Type type : source.getResultTypes()) {
        supported &= type.isIntOrFloat();
        if (type.isIntOrFloat()) bits = std::max(bits, type.getIntOrFloatBitWidth());
      }
      for (Value value : source.getInitArgs()) {
        supported &= matchPattern(value, m_Zero()) || matchPattern(value, m_PosZeroFloat());
      }
      for (Operation &op : source.getBody()->without_terminator()) {
        if (auto load = dyn_cast<memref::LoadOp>(op)) {
          supported &= load.getMemRefType().getRank() == 1;
          bits = std::max(bits, load.getType().getIntOrFloatBitWidth());
        } else {
          supported &= (op.getName().getDialectNamespace() == "arith" || op.getName().getDialectNamespace() == "math") &&
              op.getNumRegions() == 0 && isMemoryEffectFree(&op) &&
              llvm::all_of(op.getResultTypes(), [](Type type) { return type.isIntOrIndexOrFloat(); });
        }
      }
      if (!supported) continue;
      auto model = weft::x86::ResourceModel::get(source);
      bool known = matchPattern(source.getUpperBound(), m_ConstantInt(&count)) && !count.isNegative();
      auto original = weft::x86::analyzeLoop(source, model);
      if (original.unpricedOperations) continue;
      double scalarCost = iterationCost(original);
      auto outer = dyn_cast<scf::ForOp>(source->getParentOp());
      auto sourceOuterCost = outerWork(outer, source, model);
      double originalCost = known ? weft::x86::loopExecutionCost(original).total(count.getLimitedValue()) : scalarCost;
      double bestCost = originalCost;
      scf::ForOp chosen;
      scf::ForOp chosenOuter;
      SmallVector<Attribute> records;
      weft::x86::LoopResources chosenResources;
      unsigned chosenLanes = 1;
      auto eraseChosen = [&]() {
        if (chosenOuter) b.eraseOp(chosenOuter);
        else if (chosen) b.eraseOp(chosen);
        chosen = {}; chosenOuter = {};
      };
      if (outer && sourceOuterCost) {
        APInt outputs;
        bool knownOutputs = matchPattern(outer.getUpperBound(), m_ConstantInt(&outputs)) && outputs.isStrictlyPositive();
        for (unsigned lanes = model.vectorBits / llvm::PowerOf2Ceil(bits); lanes >= 2; lanes /= 2) {
          if (knownOutputs && outputs.getLimitedValue() < lanes) continue;
          auto candidate = weft::x86::packetizeContractionOutputs(b, source, lanes);
          if (failed(candidate)) continue;
          auto resources = weft::x86::analyzeLoop(candidate->reduction, model);
          auto outerResources = weft::x86::analyzeLoop(candidate->outer, model);
          resources.peak.vectors = std::max(resources.peak.vectors, outerResources.peak.vectors);
          resources.peak.predicates = std::max(resources.peak.predicates, outerResources.peak.predicates);
          auto preparation = outerWork(candidate->outer, candidate->reduction, model);
          if (!preparation) { b.eraseOp(candidate->outer); continue; }
          double cost = iterationCost(resources) / lanes;
          if (known) {
            double packetCost = weft::x86::loopExecutionCost(resources).total(count.getLimitedValue()) + *preparation;
            if (knownOutputs)
              cost = ((outputs.getLimitedValue() / lanes) * packetCost +
                  (outputs.getLimitedValue() % lanes) * (originalCost + *sourceOuterCost)) /
                  outputs.getLimitedValue() - *sourceOuterCost;
            else cost = packetCost / lanes - *sourceOuterCost;
          }
          records.push_back(b.getDictionaryAttr({
            b.getNamedAttr("packet_axis", b.getStringAttr("output")),
            b.getNamedAttr("lanes", b.getI64IntegerAttr(lanes)),
            b.getNamedAttr("estimated_cost", b.getF64FloatAttr(cost)),
            b.getNamedAttr("preparation_and_store_cost", b.getF64FloatAttr(*preparation)),
            b.getNamedAttr("resources", weft::x86::resourceAttributes(b, resources)),
            b.getNamedAttr("decision", b.getStringAttr(resources.unpricedOperations ? "unpriced-operation" :
                model.fits(resources.peak) ? "legal" : "resource-limit"))}));
          if (!resources.unpricedOperations && model.fits(resources.peak) && cost < bestCost) {
            eraseChosen();
            chosenOuter = candidate->outer; chosen = candidate->reduction;
            chosenLanes = lanes; chosenResources = resources; bestCost = cost;
          } else b.eraseOp(candidate->outer);
        }
      }
      for (unsigned lanes = model.vectorBits / llvm::PowerOf2Ceil(bits); lanes >= 2; lanes /= 2) {
        if (known && count.getLimitedValue() < lanes) continue;
        auto candidate = packetize(b, source, lanes);
        eraseDeadPure(b, candidate);
        auto resources = weft::x86::analyzeLoop(candidate, model);
        double finish = llvm::Log2_64(lanes) * source.getNumResults() *
            (isa<FloatType>(source.getResult(0).getType()) ? 3.0 : 1.0);
        double cost = iterationCost(resources) / lanes;
        if (known) cost = weft::x86::loopExecutionCost(resources).total(count.getLimitedValue() / lanes) +
            weft::x86::loopExecutionCost(original).total(count.getLimitedValue() % lanes) + finish;
        records.push_back(b.getDictionaryAttr({
          b.getNamedAttr("packet_axis", b.getStringAttr("reduction")),
          b.getNamedAttr("lanes", b.getI64IntegerAttr(lanes)),
          b.getNamedAttr("estimated_cost", b.getF64FloatAttr(cost)),
          b.getNamedAttr("result_conversion_cost", b.getF64FloatAttr(finish)),
          b.getNamedAttr("resources", weft::x86::resourceAttributes(b, resources)),
          b.getNamedAttr("decision", b.getStringAttr(resources.unpricedOperations ? "unpriced-operation" :
              model.fits(resources.peak) ? "legal" : "resource-limit"))}));
        if (!resources.unpricedOperations && model.fits(resources.peak) && cost < bestCost) {
          eraseChosen();
          chosen = candidate; chosenLanes = lanes; chosenResources = resources; bestCost = cost;
        } else b.eraseOp(candidate);
      }
      auto function = source->getParentOfType<func::FuncOp>();
      SmallVector<Attribute> schedules;
      if (auto previous = function->getAttrOfType<ArrayAttr>("weft.x86.array_schedule")) llvm::append_range(schedules, previous);
      NamedAttrList record;
      record.assign(weft::x86::resourceAttributes(b, chosen ? chosenResources : original).getValue());
      record.set("resource_decision", b.getStringAttr(chosen || model.fits(original.peak) ?
          "within-budget" : "original-requires-llvm-allocation"));
      record.set("kind", b.getStringAttr("contraction"));
      record.set("packet_axis", b.getStringAttr(chosenOuter ? "output" : "reduction"));
      record.set("lanes", b.getI64IntegerAttr(chosenLanes));
      record.set("packet_candidates", b.getArrayAttr(records));
      record.set("packet_estimated_cost", b.getF64FloatAttr(bestCost));
      record.set("cost_basis", b.getStringAttr(known ? "complete-reduction-with-tail-and-conversion" : "steady-state-per-product; startup-recorded-separately"));
      record.set("tail", b.getStringAttr("scalar-in-source-domain"));
      unsigned id = schedules.size(); schedules.push_back(record.getDictionary(&getContext()));
      function->setAttr("weft.x86.array_schedule", b.getArrayAttr(schedules));
      if (!chosen) continue;
      chosen->setAttr("weft.x86.array_reduction", b.getUnitAttr());
      chosen->setAttr("weft.x86.array_schedule_id", b.getI32IntegerAttr(id));
      chosen->setAttr("weft.x86.packet_lanes", b.getI64IntegerAttr(chosenLanes));
      if (auto scope = source->getAttr("weft.x86.array_scope")) chosen->setAttr("weft.x86.array_scope", scope);
      if (chosenOuter) {
        chosen->setAttr("weft.x86.output_packet", b.getUnitAttr());
        outer.setLowerBound(chosenOuter.getUpperBound());
        continue;
      }
      b.setInsertionPointAfter(chosen);
      SmallVector<Value> initial;
      for (auto [i, value] : llvm::enumerate(chosen.getResults()))
        initial.push_back(vector::ReductionOp::create(b, source.getLoc(), i == 0 ? vector::CombiningKind::ADD : vector::CombiningKind::OR, value));
      source.setLowerBound(chosen.getUpperBound()); source.getInitArgsMutable().assign(initial);
    }
  }
};
} // namespace
std::unique_ptr<Pass> weft::createX86ContractionSchedulePass() { return std::make_unique<ScheduleContractions>(); }
