#include "Weft/Target/X86Compiler.h"
#include "Resources.h"
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
        supported &= matchPattern(value, m_Zero());
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
      double scalarCost = weft::x86::analyzeLoop(source, model).work + 3;
      double bestCost = known ? scalarCost * count.getLimitedValue() : scalarCost;
      scf::ForOp chosen;
      SmallVector<Attribute> records;
      weft::x86::LoopResources chosenResources;
      unsigned chosenLanes = 1;
      for (unsigned lanes = model.vectorBits / llvm::PowerOf2Ceil(bits); lanes >= 2; lanes /= 2) {
        if (known && count.getLimitedValue() < lanes) continue;
        auto candidate = packetize(b, source, lanes);
        auto resources = weft::x86::analyzeLoop(candidate, model);
        double cost = double(resources.work + 3) / lanes;
        if (known) cost = double(count.getLimitedValue() / lanes) * (resources.work + 3) +
            (count.getLimitedValue() % lanes) * scalarCost + llvm::Log2_64(lanes) * source.getNumResults();
        records.push_back(b.getDictionaryAttr({
          b.getNamedAttr("lanes", b.getI64IntegerAttr(lanes)),
          b.getNamedAttr("estimated_cost", b.getF64FloatAttr(cost)),
          b.getNamedAttr("resources", weft::x86::resourceAttributes(b, resources)),
          b.getNamedAttr("decision", b.getStringAttr(model.fits(resources.peak) ? "legal" : "resource-limit"))}));
        if (model.fits(resources.peak) && cost < bestCost) {
          if (chosen) b.eraseOp(chosen);
          chosen = candidate; chosenLanes = lanes; chosenResources = resources; bestCost = cost;
        } else b.eraseOp(candidate);
      }
      auto function = source->getParentOfType<func::FuncOp>();
      SmallVector<Attribute> schedules;
      if (auto previous = function->getAttrOfType<ArrayAttr>("weft.x86.array_schedule")) llvm::append_range(schedules, previous);
      NamedAttrList record;
      if (chosen) record.assign(weft::x86::resourceAttributes(b, chosenResources).getValue());
      record.set("kind", b.getStringAttr("contraction"));
      record.set("lanes", b.getI64IntegerAttr(chosenLanes));
      record.set("packet_candidates", b.getArrayAttr(records));
      record.set("packet_estimated_cost", b.getF64FloatAttr(bestCost));
      record.set("tail", b.getStringAttr("scalar-in-source-domain"));
      unsigned id = schedules.size(); schedules.push_back(record.getDictionary(&getContext()));
      function->setAttr("weft.x86.array_schedule", b.getArrayAttr(schedules));
      if (!chosen) continue;
      chosen->setAttr("weft.x86.array_reduction", b.getUnitAttr());
      chosen->setAttr("weft.x86.array_schedule_id", b.getI32IntegerAttr(id));
      chosen->setAttr("weft.x86.packet_lanes", b.getI64IntegerAttr(chosenLanes));
      if (auto scope = source->getAttr("weft.x86.array_scope")) chosen->setAttr("weft.x86.array_scope", scope);
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
