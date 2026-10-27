#include "Analysis/Resources.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace weft::x86 {
ResourceModel ResourceModel::get(Operation *op) {
  auto function = op->getParentOfType<func::FuncOp>();
  unsigned bits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits").getInt();
  unsigned registers = function->getAttrOfType<IntegerAttr>("weft.x86.vector_registers").getInt();
  return {bits, registers, bits == 512 ? 7u : 0u};
}
RegisterUsage ResourceModel::storage(Type type) const {
  uint64_t lanes = 0;
  if (auto vector = dyn_cast<VectorType>(type)) {
    lanes = vector.getNumElements(); type = vector.getElementType();
  }
  if (!lanes) return {isa<FloatType>(type) ? 1u : 0u, 0};
  if (!type.isIntOrIndexOrFloat()) return {};
  if (type.isInteger(1) && predicateRegisters) return {0, llvm::divideCeil(lanes, uint64_t(64))};
  unsigned width = type.isIndex() ? 64 : std::max(8u, type.getIntOrFloatBitWidth());
  return {llvm::divideCeil(lanes * width, uint64_t(vectorBits)), 0};
}
static void maximize(RegisterUsage &a, RegisterUsage b) {
  a.vectors = std::max(a.vectors, b.vectors);
  a.predicates = std::max(a.predicates, b.predicates);
}
LoopExecutionCost loopExecutionCost(const LoopResources &resources) {
  double period = std::max(resources.issueUnits, resources.carriedPath) + 3;
  return {period, std::max(0.0, resources.criticalPath - period)};
}
static std::optional<InstructionCost> operationCost(Operation *op, const ResourceModel &model) {
  auto bytes = [](Type type) -> uint64_t {
    uint64_t elements = 1;
    if (auto vector = dyn_cast<VectorType>(type)) {
      if (vector.isScalable()) return 0;
      elements = vector.getNumElements(); type = vector.getElementType();
    }
    if (!type.isIntOrIndexOrFloat()) return 0;
    return elements * llvm::divideCeil(type.isIndex() ? 64u : type.getIntOrFloatBitWidth(), 8u);
  };
  if (op->hasTrait<OpTrait::IsTerminator>() ||
      isa<arith::ConstantOp, ub::PoisonOp, arith::BitcastOp, vector::BitCastOp, vector::ShapeCastOp>(op))
    return InstructionCost{};
  uint64_t packets = 1;
  for (Type type : op->getOperandTypes()) {
    auto usage = model.storage(type);
    packets = std::max(packets, usage.vectors + usage.predicates);
  }
  for (Type type : op->getResultTypes()) {
    auto usage = model.storage(type);
    packets = std::max(packets, usage.vectors + usage.predicates);
  }
  auto cost = [&](double issues, double latency) {
    return InstructionCost{issues * packets, latency, 0, 0};
  };
  if (isa<memref::LoadOp, vector::LoadOp>(op))
    return InstructionCost{double(packets), 4, bytes(op->getResult(0).getType()), 0};
  if (isa<memref::StoreOp, vector::StoreOp>(op))
    return InstructionCost{double(packets), 1, 0, bytes(op->getOperand(0).getType())};
  if (auto gather = dyn_cast<vector::GatherOp>(op)) {
    auto type = cast<VectorType>(gather.getResult().getType());
    return InstructionCost{double(type.getNumElements()), 4, bytes(type), 0};
  }
  if (auto scatter = dyn_cast<vector::ScatterOp>(op)) {
    auto type = cast<VectorType>(scatter.getValueToStore().getType());
    return InstructionCost{double(type.getNumElements()), 1, 0, bytes(type)};
  }
  if (isa<arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp>(op)) {
    APInt divisor;
    Value denominator = op->getOperand(1);
    while (auto broadcast = denominator.getDefiningOp<vector::BroadcastOp>()) denominator = broadcast.getSource();
    bool constant = matchPattern(denominator, m_ConstantInt(&divisor));
    DenseIntElementsAttr splat;
    if (!constant && matchPattern(denominator, m_Constant(&splat)) && splat.isSplat()) {
      divisor = splat.getSplatValue<APInt>();
      constant = true;
    }
    if (isa<arith::DivUIOp, arith::RemUIOp>(op) &&
        constant && !divisor.isZero())
      return divisor.isPowerOf2() ? cost(1, 1) : cost(4, 6);
    // There is no general SIMD integer divide. Charge its scalar lanes, rather
    // than pretending one wide IR operation is one machine instruction.
    auto type = dyn_cast<VectorType>(op->getResult(0).getType());
    return InstructionCost{24.0 * (type ? type.getNumElements() : 1), 24, 0, 0};
  }
  if (isa<arith::DivFOp>(op)) return cost(8, 12);
  if (isa<math::SqrtOp>(op)) return cost(8, 12);
  if (isa<math::RsqrtOp>(op)) return cost(16, 24);
  if (auto absolute = dyn_cast<math::AbsIOp>(op)) {
    auto vector = dyn_cast<VectorType>(absolute.getType());
    if (!vector) return cost(2, 2); // scalar negation and conditional selection
    // AVX2 supplies byte/word/dword absolute value; qwords need sign, xor and
    // subtraction. AVX-512 provides the native qword form as well.
    return vector.getElementTypeBitWidth() == 64 && model.vectorBits < 512 ?
        cost(3, 3) : cost(1, 1);
  }
  if (isa<math::FloorOp, math::CeilOp, math::TruncOp, math::RoundEvenOp>(op))
    return cost(1, 3);
  if (isa<arith::MulIOp>(op)) return cost(1, 3);
  if (isa<arith::MulFOp, math::FmaOp>(op)) return cost(1, 4);
  if (isa<arith::AddFOp, arith::SubFOp>(op)) return cost(1, 3);
  if (isa<arith::MinNumFOp, arith::MaxNumFOp, arith::MinimumFOp, arith::MaximumFOp>(op))
    return cost(2, 3);
  if (isa<arith::AddIOp, arith::SubIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
          arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::CmpIOp, arith::CmpFOp,
          arith::SelectOp, arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp,
          arith::NegFOp, math::AbsFOp>(op)) return cost(1, 1);
  if (isa<arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
          arith::IndexCastOp, arith::IndexCastUIOp>(op)) return cost(1, 3);
  if (isa<vector::BroadcastOp, vector::ExtractOp, vector::InsertOp,
          vector::ExtractStridedSliceOp, vector::InsertStridedSliceOp, vector::ShuffleOp>(op))
    return cost(1, 3);
  if (auto reduction = dyn_cast<vector::ReductionOp>(op)) {
    auto type = cast<VectorType>(reduction.getVector().getType());
    double depth = llvm::Log2_64_Ceil(type.getNumElements());
    return cost(depth, depth * (isa<FloatType>(type.getElementType()) ? 3 : 1));
  }
  return std::nullopt;
}
std::optional<InstructionCost> instructionCost(Operation *op, const ResourceModel &model) {
  auto result = operationCost(op, model);
  if (!result) return {};
  llvm::SmallPtrSet<Value, 4> literals;
  for (Value value : op->getOperands())
    if (value.getDefiningOp<arith::ConstantOp>() && literals.insert(value).second) {
      auto storage = model.storage(value.getType());
      result->issues += storage.vectors + storage.predicates;
    }
  return result;
}
static LoopResources analyzeBlock(Block *body, Operation *scope,
                                 const ResourceModel &model, Liveness &liveness,
                                 ValueRange carries) {
  LoopResources result;
  llvm::SetVector<Value> resident;
  // A value can remain resident in an enclosing loop without being captured by
  // this loop. Include these intervals as well as direct local captures.
  auto function = dyn_cast<func::FuncOp>(scope);
  if (!function) function = scope->getParentOfType<func::FuncOp>();
  for (Operation *at = scope; at && at != function.getOperation(); at = at->getParentOp()) {
    auto *info = liveness.getLiveness(at->getBlock());
    if (!info) continue;
    for (Value value : info->currentlyLiveValues(at)) {
      Operation *owner = value.getDefiningOp();
      if (!owner) owner = value.getParentBlock()->getParentOp();
      if (owner == at || at->isProperAncestor(owner)) continue;
      if (!liveness.isDeadAfter(value, at)) resident.insert(value);
    }
  }
  for (Operation &op : *body)
    for (Value value : op.getOperands())
      if (value.getParentBlock() != body) resident.insert(value);
  RegisterUsage live;
  for (Value value : resident)
    if (!value.getDefiningOp<arith::ConstantOp>() && !value.getDefiningOp<ub::PoisonOp>())
      live += model.storage(value.getType());
  result.surrounding = live;
  DenseMap<Value, unsigned> lastUse;
  for (auto [i, op] : llvm::enumerate(*body)) {
    if (auto estimate = instructionCost(&op, model)) {
      result.issueUnits += estimate->issues;
      result.readBytes += estimate->readBytes;
      result.writtenBytes += estimate->writtenBytes;
    } else ++result.unpricedOperations;
    result.permutations += isa<vector::ShuffleOp, vector::ExtractStridedSliceOp,
                               vector::InsertStridedSliceOp, vector::BroadcastOp>(op);
    op.walk([&](Operation *nested) {
      for (Value value : nested->getOperands())
        if (value.getParentBlock() == body) lastUse[value] = i;
    });
  }
  DenseMap<Value, RegisterUsage> costs;
  for (Value value : body->getArguments()) {
    if (!lastUse.count(value)) continue;
    auto cost = model.storage(value.getType());
    costs[value] = cost; live += cost;
  }
  maximize(result.peak, live);
  for (auto [i, op] : llvm::enumerate(*body)) {
    uint64_t issues = 0;
    for (Value value : op.getResults()) {
      // Poison owns no physical bits; constants are charged where rematerialized
      // below. Values that define lanes, freezes and carries remain resident.
      auto cost = isa<ub::PoisonOp, arith::ConstantOp>(op) ? RegisterUsage{} : model.storage(value.getType());
      costs[value] = cost; live += cost;
      issues = std::max(issues, cost.vectors + cost.predicates);
    }
    if (!op.hasTrait<OpTrait::IsTerminator>() && !isa<arith::ConstantOp>(op))
      result.work += std::max(uint64_t(1), issues);
    bool load = isa<memref::LoadOp, vector::LoadOp, vector::GatherOp>(op);
    bool store = isa<memref::StoreOp, vector::StoreOp, vector::ScatterOp>(op);
    result.loads += load; result.stores += store;
    bool cast = isa<arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp,
                    arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
                    arith::FPToUIOp, arith::IndexCastOp, arith::IndexCastUIOp>(op);
    result.conversions += cast;
    // Conversion and indexed access legalization can need a temporary carrier.
    // Charge this at the operation, not an unconditional reserve on every loop.
    RegisterUsage scratch;
    // Constants have no state or snapshot identity: LLVM can rematerialize them
    // at use (zero idiom, immediate, or constant-pool load). Charge their carriers
    // locally instead of retaining every differently shaped literal across loops.
    llvm::SmallPtrSet<Value, 4> literals;
    for (Value value : op.getOperands())
      if (value.getDefiningOp<arith::ConstantOp>() && literals.insert(value).second) {
        scratch += model.storage(value.getType());
        result.work += model.storage(value.getType()).vectors;
      }
    if (cast || isa<vector::GatherOp, vector::ScatterOp>(op)) ++scratch.vectors;
    if (isa<vector::GatherOp, vector::ScatterOp>(op)) {
      if (model.predicateRegisters) scratch.predicates = 1;
      else ++scratch.vectors;
      auto type = dyn_cast<VectorType>(op.getOperand(op.getNumOperands() - 1).getType());
      if (type) result.work += type.getNumElements();
    }
    maximize(result.scratch, scratch);
    auto at = live; at += scratch; maximize(result.peak, at);
    // Captures may be used only inside a nested region, not listed among its
    // parent operation's operands. Their actual final region use still retires
    // them here; otherwise every sequential child falsely accumulates residency.
    SmallVector<Value> retired;
    for (auto [value, cost] : costs) {
      auto last = lastUse.find(value);
      if (last != lastUse.end() && last->second == i) retired.push_back(value);
    }
    for (Value value : retired) { live -= costs.lookup(value); costs.erase(value); }
    for (Value value : op.getResults())
      if (!lastUse.count(value)) { live -= costs[value]; costs.erase(value); }
  }
  auto dependency = weft::analysis::dependencyCost(*body, carries, [&](Operation *op) {
    auto estimate = instructionCost(op, model);
    return estimate ? estimate->latency : 0;
  });
  result.criticalPath = dependency.criticalPath;
  result.carriedPath = dependency.carriedPath;
  return result;
}
LoopResources analyzeLoop(scf::ForOp loop, const ResourceModel &model) {
  Liveness liveness(loop->getParentOfType<func::FuncOp>());
  return analyzeBlock(loop.getBody(), loop, model, liveness, loop.getRegionIterArgs());
}
RegisterUsage functionRegisterPeak(func::FuncOp function, const ResourceModel &model) {
  Liveness liveness(function);
  RegisterUsage result;
  function.walk([&](Operation *scope) {
    for (Region &region : scope->getRegions())
      for (Block &body : region) {
        ValueRange carries;
        if (auto loop = dyn_cast<scf::ForOp>(scope)) carries = loop.getRegionIterArgs();
        maximize(result, analyzeBlock(&body, scope, model, liveness, carries).peak);
      }
  });
  return result;
}
DictionaryAttr resourceAttributes(OpBuilder &b, const LoopResources &r) {
  return b.getDictionaryAttr({
    b.getNamedAttr("estimated_live_vectors", b.getI64IntegerAttr(r.peak.vectors)),
    b.getNamedAttr("estimated_live_predicates", b.getI64IntegerAttr(r.peak.predicates)),
    b.getNamedAttr("surrounding_vectors", b.getI64IntegerAttr(r.surrounding.vectors)),
    b.getNamedAttr("surrounding_predicates", b.getI64IntegerAttr(r.surrounding.predicates)),
    b.getNamedAttr("scratch_vectors", b.getI64IntegerAttr(r.scratch.vectors)),
    b.getNamedAttr("work_units", b.getI64IntegerAttr(r.work)),
    b.getNamedAttr("issue_units", b.getF64FloatAttr(r.issueUnits)),
    b.getNamedAttr("critical_path", b.getF64FloatAttr(r.criticalPath)),
    b.getNamedAttr("carried_path", b.getF64FloatAttr(r.carriedPath)),
    b.getNamedAttr("loop_period", b.getF64FloatAttr(loopExecutionCost(r).period)),
    b.getNamedAttr("loop_startup", b.getF64FloatAttr(loopExecutionCost(r).startup)),
    b.getNamedAttr("read_bytes", b.getI64IntegerAttr(r.readBytes)),
    b.getNamedAttr("written_bytes", b.getI64IntegerAttr(r.writtenBytes)),
    b.getNamedAttr("permutations", b.getI64IntegerAttr(r.permutations)),
    b.getNamedAttr("unpriced_operations", b.getI64IntegerAttr(r.unpricedOperations)),
    b.getNamedAttr("loads", b.getI64IntegerAttr(r.loads)),
    b.getNamedAttr("stores", b.getI64IntegerAttr(r.stores)),
    b.getNamedAttr("conversions", b.getI64IntegerAttr(r.conversions))});
}
void updateArraySchedule(scf::ForOp loop, OpBuilder &b, ArrayRef<NamedAttribute> fields) {
  auto id = loop->getAttrOfType<IntegerAttr>("weft.x86.array_schedule_id");
  if (!id) return;
  auto function = loop->getParentOfType<func::FuncOp>();
  auto schedules = function->getAttrOfType<ArrayAttr>("weft.x86.array_schedule");
  SmallVector<Attribute> entries(schedules.begin(), schedules.end());
  NamedAttrList entry(cast<DictionaryAttr>(entries[id.getInt()]));
  for (auto field : fields) entry.set(field.getName(), field.getValue());
  entries[id.getInt()] = entry.getDictionary(b.getContext());
  function->setAttr("weft.x86.array_schedule", b.getArrayAttr(entries));
}
} // namespace weft::x86
