#include "Resources.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
RegisterUsage ResourceModel::storage(Type type, unsigned scalarLanes) const {
  uint64_t lanes = scalarLanes;
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
LoopResources analyzeLoop(scf::ForOp loop, const ResourceModel &model, unsigned scalarLanes) {
  LoopResources result;
  Block *body = loop.getBody();
  llvm::SetVector<Value> resident;
  // A value can remain resident in an enclosing loop without being captured by
  // this loop. Include these intervals as well as direct local captures.
  auto function = loop->getParentOfType<func::FuncOp>();
  Liveness liveness(function);
  for (Operation *at = loop; at && at != function.getOperation(); at = at->getParentOp()) {
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
    if (!value.getDefiningOp<arith::ConstantOp>()) live += model.storage(value.getType());
  result.surrounding = live;
  DenseMap<Value, unsigned> lastUse;
  for (auto [i, op] : llvm::enumerate(*body)) {
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
      auto cost = model.storage(value.getType(), scalarLanes);
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
      if (value.getParentBlock() != body && value.getDefiningOp<arith::ConstantOp>() && literals.insert(value).second) {
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
    for (Value value : op.getOperands())
      if (lastUse.lookup(value) == i && costs.count(value)) {
        live -= costs[value]; costs.erase(value);
      }
    for (Value value : op.getResults())
      if (!lastUse.count(value)) { live -= costs[value]; costs.erase(value); }
  }
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
