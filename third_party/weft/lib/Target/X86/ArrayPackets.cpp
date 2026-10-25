#include "Weft/Target/X86Compiler.h"
#include "ArrayPackets.h"
#include "Resources.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace {
unsigned recordSchedule(func::FuncOp function, OpBuilder &b, DictionaryAttr entry) {
  SmallVector<Attribute> records;
  if (auto old = function->getAttrOfType<ArrayAttr>("weft.x86.array_schedule"))
    llvm::append_range(records, old);
  records.push_back(entry);
  function->setAttr("weft.x86.array_schedule", b.getArrayAttr(records));
  return records.size() - 1;
}

class PacketizeArrays : public PassWrapper<PacketizeArrays, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PacketizeArrays)
  StringRef getArgument() const override { return "weft-x86-packetize-arrays"; }
  StringRef getDescription() const override {
    return "Packetize compiler-owned pointwise array loops using target SIMD resources";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.x86.array_elementwise")) loops.push_back(loop);
    });
    IRRewriter b(&getContext());
    for (auto loop : loops) {
      loop->removeAttr("weft.x86.array_elementwise");
      APInt lower, step;
      if (loop.getNumResults() || !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
          !lower.isZero() || !matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isOne()) continue;
      auto function = loop->getParentOfType<func::FuncOp>();
      auto bits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits");
      auto registers = function->getAttrOfType<IntegerAttr>("weft.x86.vector_registers");
      if (!bits || !registers) continue;
      unsigned elementBits = 8;
      bool supported = true;
      SmallVector<Operation *> body;
      for (Operation &op : loop.getBody()->without_terminator()) {
        body.push_back(&op);
        if (auto load = dyn_cast<memref::LoadOp>(op)) {
          supported &= load.getMemRefType().getRank() == 1;
          elementBits = std::max(elementBits, load.getType().isIndex() ? 64u : load.getType().getIntOrFloatBitWidth());
        } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
          supported &= store.getMemRefType().getRank() == 1 && store.getIndices()[0] == loop.getInductionVar() &&
                       store.getMemRef().getDefiningOp<memref::AllocOp>() != nullptr;
          elementBits = std::max(elementBits, store.getValue().getType().isIndex() ? 64u : store.getValue().getType().getIntOrFloatBitWidth());
        } else {
          StringRef dialect = op.getName().getDialectNamespace();
          supported &= (dialect == "arith" || dialect == "math") && op.getNumRegions() == 0 &&
                       isMemoryEffectFree(&op) && llvm::all_of(op.getResultTypes(), [](Type type) { return type.isIntOrIndexOrFloat(); });
        }
      }
      if (!supported) continue;
      loop->setAttr("weft.x86.array_pointwise", b.getUnitAttr());
      auto model = weft::x86::ResourceModel::get(loop);
      SmallVector<Attribute> candidateRecords;
      APInt count;
      bool knownCount = matchPattern(loop.getUpperBound(), m_ConstantInt(&count)) && !count.isNegative();
      auto original = weft::x86::analyzeLoop(loop, model);
      auto scalarCost = weft::x86::loopExecutionCost(original);
      // Cost selection uses completed physical bodies. A scalar-body estimate
      // cannot predict which accesses become loads, gathers or broadcasts, nor
      // the conversion scratch and constant packets those choices introduce.
      double bestCost = knownCount ? scalarCost.total(count.getLimitedValue()) : scalarCost.period;
      unsigned rejected = 0;
      scf::ForOp chosen;
      weft::x86::LoopResources chosenResources;
      int64_t chosenLanes = 1;
      unsigned chosenRecord = 0;
      for (int64_t lanes = bits.getInt() / llvm::PowerOf2Ceil(elementBits); lanes >= 2; lanes /= 2) {
        Location loc = loop.getLoc();
        b.setInsertionPoint(loop);
        Value width = arith::ConstantIndexOp::create(b, loc, lanes);
        Value remainder = b.createOrFold<arith::RemUIOp>(loc, loop.getUpperBound(), width);
        Value end = b.createOrFold<arith::SubIOp>(loc, loop.getUpperBound(), remainder);
        auto packet = scf::ForOp::create(b, loc, loop.getLowerBound(), end, width);
        packet->setAttr("weft.x86.array_pointwise", b.getUnitAttr());
        if (auto scope = loop->getAttr("weft.x86.array_scope"))
          packet->setAttr("weft.x86.array_scope", scope);
        packet->setAttr("weft.x86.packet_lanes", b.getI64IntegerAttr(lanes));
        b.setInsertionPointToStart(packet.getBody());
        IRMapping mapping;
        auto vectorType = [&](Type type) { return VectorType::get({lanes}, type); };
        auto broadcast = [&](Value value) -> Value {
          return isa<VectorType>(value.getType()) ? value : Value(vector::BroadcastOp::create(b, loc, vectorType(value.getType()), value));
        };
        SmallVector<int64_t> sequence;
        for (int64_t i = 0; i < lanes; ++i) sequence.push_back(i);
        auto indexType = vectorType(b.getIndexType()), maskType = vectorType(b.getI1Type());
        Value offsets = arith::ConstantOp::create(b, loc, indexType, DenseIntElementsAttr::get(indexType, sequence));
        Value indices = arith::AddIOp::create(b, loc, broadcast(packet.getInductionVar()), offsets);
        Value zero = arith::ConstantIndexOp::create(b, loc, 0);
        Value active = arith::ConstantOp::create(b, loc, maskType, DenseElementsAttr::get(maskType, b.getBoolAttr(true)));
        mapping.map(loop.getInductionVar(), indices);
        for (Operation *op : body) {
          if (auto load = dyn_cast<memref::LoadOp>(op)) {
            Value index = mapping.lookupOrDefault(load.getIndices()[0]), value;
            if (!isa<VectorType>(index.getType())) value = memref::LoadOp::create(b, loc, load.getMemRef(), index);
            else if (load.getIndices()[0] == loop.getInductionVar() && !load.getType().isInteger(1))
              value = vector::LoadOp::create(b, loc, vectorType(load.getType()), load.getMemRef(), packet.getInductionVar());
            else {
              Value fill = arith::ConstantOp::create(b, loc, vectorType(load.getType()), b.getZeroAttr(vectorType(load.getType())));
              value = vector::GatherOp::create(b, loc, vectorType(load.getType()), load.getMemRef(), ValueRange{zero}, index, active, fill);
            }
            mapping.map(load.getResult(), value);
          } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
            Value value = broadcast(mapping.lookupOrDefault(store.getValue()));
            // MemRef<i1> uses a byte for each scalar. Gather/scatter retain that
            // per-element storage relation; a packed vector<i1> load/store does not.
            if (store.getValue().getType().isInteger(1))
              vector::ScatterOp::create(b, loc, Type{}, store.getMemRef(), ValueRange{zero}, indices, active, value);
            else vector::StoreOp::create(b, loc, value, store.getMemRef(), packet.getInductionVar());
          } else {
            SmallVector<Value> operands;
            bool shaped = false;
            for (Value value : op->getOperands()) {
              Value mapped = mapping.lookupOrDefault(value); operands.push_back(mapped);
              shaped |= isa<VectorType>(mapped.getType());
            }
            if (shaped) for (Value &value : operands) value = broadcast(value);
            OperationState state(loc, op->getName()); state.addAttributes(op->getAttrs()); state.addOperands(operands);
            for (Type type : op->getResultTypes()) state.addTypes(shaped ? Type(vectorType(type)) : type);
            Operation *copy = b.create(state);
            mapping.map(op->getResults(), copy->getResults());
          }
        }
        for (Operation *op = packet.getBody()->getTerminator()->getPrevNode(); op;) {
          Operation *previous = op->getPrevNode();
          if (isOpTriviallyDead(op)) b.eraseOp(op);
          op = previous;
        }
        auto actual = weft::x86::analyzeLoop(packet, model);
        auto execution = weft::x86::loopExecutionCost(actual);
        double setup = 0;
        // The packet endpoint is invariant over this traversal. Static endpoint
        // arithmetic folds above; dynamic quotient/remainder preparation is
        // charged once, not once per element.
        for (Value value : {remainder, end})
          if (Operation *op = value.getDefiningOp(); op && !isa<arith::ConstantOp>(op)) {
            auto price = weft::x86::instructionCost(op, model);
            if (price) setup += price->issues;
            else ++actual.unpricedOperations;
          }
        double work = execution.period / lanes;
        if (knownCount) {
          uint64_t size = count.getLimitedValue();
          work = setup + execution.total(size / lanes) + scalarCost.total(size % lanes);
        }
        bool complete = !actual.unpricedOperations && !original.unpricedOperations;
        bool fits = model.fits(actual.peak);
        unsigned record = candidateRecords.size();
        candidateRecords.push_back(b.getDictionaryAttr({
          b.getNamedAttr("lanes", b.getI64IntegerAttr(lanes)),
          b.getNamedAttr("estimated_cost", b.getF64FloatAttr(work)),
          b.getNamedAttr("setup_cost", b.getF64FloatAttr(setup)),
          b.getNamedAttr("startup_cost", b.getF64FloatAttr(execution.startup)),
          b.getNamedAttr("resources", weft::x86::resourceAttributes(b, actual)),
          b.getNamedAttr("decision", b.getStringAttr(!complete ? "unpriced-operation" :
              !fits ? "physical-resource-limit" : "legal"))}));
        if (!fits) ++rejected;
        if (complete && fits && work < bestCost) {
          if (chosen) b.eraseOp(chosen);
          chosen = packet; chosenResources = actual;
          chosenLanes = lanes; chosenRecord = record; bestCost = work;
        } else b.eraseOp(packet);
      }
      if (chosen) {
        loop.setLowerBound(chosen.getUpperBound());
        NamedAttrList checked(cast<DictionaryAttr>(candidateRecords[chosenRecord]));
        checked.set("decision", b.getStringAttr("selected"));
        candidateRecords[chosenRecord] = checked.getDictionary(&getContext());
        NamedAttrList entry(weft::x86::resourceAttributes(b, chosenResources));
        entry.set("kind", b.getStringAttr("pointwise"));
        entry.set("lanes", b.getI64IntegerAttr(chosenLanes));
        entry.set("element_bits", b.getI64IntegerAttr(elementBits));
        entry.set("vector_register_capacity", registers);
        entry.set("predicate_register_capacity", b.getI64IntegerAttr(model.predicateRegisters));
        entry.set("rejected_resource_candidates", b.getI64IntegerAttr(rejected));
        entry.set("candidate_count", b.getI64IntegerAttr(candidateRecords.size()));
        entry.set("cost_basis", b.getStringAttr(knownCount ? "complete-physical-domain-with-tail-and-startup" : "physical-steady-state-per-element; startup-recorded-separately"));
        entry.set("candidates", b.getArrayAttr(candidateRecords));
        entry.set("estimated_selected_cost", b.getF64FloatAttr(bestCost));
        unsigned schedule = recordSchedule(function, b, entry.getDictionary(&getContext()));
        for (auto part : {loop, chosen}) part->setAttr("weft.x86.array_schedule_id", b.getI32IntegerAttr(schedule));
      } else {
        NamedAttrList entry(weft::x86::resourceAttributes(b, original));
        entry.set("kind", b.getStringAttr("scalar-pointwise"));
        entry.set("lanes", b.getI64IntegerAttr(1));
        entry.set("candidate_count", b.getI64IntegerAttr(candidateRecords.size()));
        entry.set("estimated_selected_cost", b.getF64FloatAttr(bestCost));
        entry.set("cost_complete", b.getBoolAttr(!original.unpricedOperations));
        entry.set("rejected_resource_candidates", b.getI64IntegerAttr(rejected));
        entry.set("candidates", b.getArrayAttr(candidateRecords));
        unsigned schedule = recordSchedule(function, b, entry.getDictionary(&getContext()));
        loop->setAttr("weft.x86.array_schedule_id", b.getI32IntegerAttr(schedule));
      }
    }
  }
};
} // namespace
std::unique_ptr<Pass> weft::createX86ArrayPacketPass() { return std::make_unique<PacketizeArrays>(); }

std::pair<Value, Value> weft::x86::reduceArrayPackets(
    OpBuilder &b, Location loc, Value data, Value valid, Value base, Value stride,
    Value count, Value identity, vector::CombiningKind kind, StringRef scalarOperation,
    unsigned vectorBits, Attribute sourceScope) {
  Type elem = identity.getType();
  int64_t lanes = vectorBits / elem.getIntOrFloatBitWidth();
  auto packetType = VectorType::get({lanes}, elem);
  auto maskType = VectorType::get({lanes}, b.getI1Type());
  auto indexType = VectorType::get({lanes}, b.getIndexType());
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value width = arith::ConstantIndexOp::create(b, loc, lanes);
  Value split = arith::SubIOp::create(b, loc, count, arith::RemUIOp::create(b, loc, count, width));
  Value initial = vector::BroadcastOp::create(b, loc, packetType, identity);
  Value all = arith::ConstantOp::create(b, loc, maskType, DenseElementsAttr::get(maskType, b.getBoolAttr(true)));
  Value none = arith::ConstantOp::create(b, loc, maskType, b.getZeroAttr(maskType));
  auto apply = [&](Value left, Value right) -> Value {
    OperationState state(loc, scalarOperation); state.addOperands({left, right}); state.addTypes(left.getType());
    return b.create(state)->getResult(0);
  };
  auto loop = scf::ForOp::create(b, loc, zero, split, width, ValueRange{initial, none});
  loop->setAttr("weft.x86.array_reduction", b.getUnitAttr());
  loop->setAttr("weft.x86.packet_lanes", b.getI64IntegerAttr(lanes));
  unsigned schedule = recordSchedule(loop->getParentOfType<func::FuncOp>(), b, b.getDictionaryAttr({
      b.getNamedAttr("kind", b.getStringAttr("reduction")),
      b.getNamedAttr("lanes", b.getI64IntegerAttr(lanes)),
      b.getNamedAttr("element_bits", b.getI64IntegerAttr(elem.getIntOrFloatBitWidth())),
      b.getNamedAttr("partial_vectors", b.getI64IntegerAttr(2)),
      b.getNamedAttr("tail", b.getStringAttr("scalar-in-source-domain"))}));
  {
    OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(loop.getBody());
    SmallVector<int64_t> sequence;
    for (int64_t lane = 0; lane < lanes; ++lane) sequence.push_back(lane);
    Value offsets = arith::ConstantOp::create(b, loc, indexType, DenseIntElementsAttr::get(indexType, sequence));
    auto broadcast = [&](Value value) -> Value { return vector::BroadcastOp::create(b, loc, indexType, value); };
    Value indices = arith::AddIOp::create(b, loc, broadcast(loop.getInductionVar()), offsets);
    indices = arith::AddIOp::create(b, loc, broadcast(base), arith::MulIOp::create(b, loc, indices, broadcast(stride)));
    APInt step;
    Value input, bytes;
    auto byteType = VectorType::get({lanes}, b.getI8Type());
    if (matchPattern(stride, m_ConstantInt(&step)) && step.isOne()) {
      Value start = arith::AddIOp::create(b, loc, base, loop.getInductionVar());
      input = vector::LoadOp::create(b, loc, packetType, data, start);
      bytes = vector::LoadOp::create(b, loc, byteType, valid, start);
    } else {
      input = vector::GatherOp::create(b, loc, packetType, data, ValueRange{zero}, indices, all, initial);
      Value fill = arith::ConstantOp::create(b, loc, byteType, b.getZeroAttr(byteType));
      bytes = vector::GatherOp::create(b, loc, byteType, valid, ValueRange{zero}, indices, all, fill);
    }
    Value active = arith::TruncIOp::create(b, loc, maskType, bytes);
    input = arith::SelectOp::create(b, loc, active, input, initial);
    Value combined = apply(loop.getRegionIterArgs()[0], input);
    Value any = arith::OrIOp::create(b, loc, loop.getRegionIterArgs()[1], active);
    scf::YieldOp::create(b, loc, ValueRange{combined, any});
  }
  Value partial = vector::ReductionOp::create(b, loc, kind, loop.getResult(0));
  Value any = vector::ReductionOp::create(b, loc, vector::CombiningKind::OR, loop.getResult(1));
  auto tail = scf::ForOp::create(b, loc, split, count, one, ValueRange{partial, any});
  tail->setAttr("weft.x86.array_reduction", b.getUnitAttr());
  for (auto part : {loop, tail}) {
    part->setAttr("weft.x86.array_schedule_id", b.getI32IntegerAttr(schedule));
    part->setAttr("weft.x86.array_scope", sourceScope);
  }
  {
    OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(tail.getBody());
    Value index = arith::AddIOp::create(b, loc, base, arith::MulIOp::create(b, loc, tail.getInductionVar(), stride));
    Value byte = memref::LoadOp::create(b, loc, valid, index);
    Value active = arith::TruncIOp::create(b, loc, b.getI1Type(), byte);
    Value input = memref::LoadOp::create(b, loc, data, index);
    input = arith::SelectOp::create(b, loc, active, input, identity);
    scf::YieldOp::create(b, loc, ValueRange{apply(tail.getRegionIterArgs()[0], input),
        arith::OrIOp::create(b, loc, tail.getRegionIterArgs()[1], active)});
  }
  return {tail.getResult(0), tail.getResult(1)};
}
