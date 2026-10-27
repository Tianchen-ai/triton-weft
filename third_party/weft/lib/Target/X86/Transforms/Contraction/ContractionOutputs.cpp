#include "Transforms/Contraction/ContractionOutputs.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
namespace {
Value storageRoot(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>()) value = cast.getSource();
  return value;
}

// Coefficient within an aligned output packet. Div/rem of its induction
// variable by a complete row width cannot cross a row inside that packet.
std::optional<int64_t> coefficient(Value value, scf::ForOp outer,
                                    scf::ForOp reduction, unsigned lanes,
                                    unsigned depth = 0) {
  if (value == outer.getInductionVar()) return 1;
  if (value == reduction.getInductionVar()) return 0;
  Operation *op = value.getDefiningOp();
  if (!op) {
    auto *owner = value.getParentBlock()->getParentOp();
    return outer->isProperAncestor(owner) ? std::nullopt : std::optional<int64_t>(0);
  }
  if (!outer->isProperAncestor(op)) return 0;
  if (depth >= 32) return {};
  if (isa<arith::ConstantOp>(op)) return 0;
  auto input = [&](Value operand) { return coefficient(operand, outer, reduction, lanes, depth + 1); };
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    auto a = input(op->getOperand(0)), c = input(op->getOperand(1));
    int64_t result;
    if (a && c && !(isa<arith::AddIOp>(op) ? llvm::AddOverflow(*a, *c, result) : llvm::SubOverflow(*a, *c, result)))
      return result;
  }
  if (auto multiply = dyn_cast<arith::MulIOp>(op)) {
    if (input(multiply.getLhs()) == std::optional<int64_t>(0) &&
        input(multiply.getRhs()) == std::optional<int64_t>(0)) return 0;
    APInt scale;
    for (unsigned side = 0; side < 2; ++side)
      if (matchPattern(multiply->getOperand(side), m_ConstantInt(&scale)) && scale.isSignedIntN(32))
        if (auto other = input(multiply->getOperand(1 - side))) {
          int64_t result;
          if (!llvm::MulOverflow(*other, scale.getSExtValue(), result)) return result;
        }
  }
  if (isa<arith::DivUIOp, arith::RemUIOp>(op)) {
    auto a = input(op->getOperand(0)), c = input(op->getOperand(1));
    if (a && c && *a == 0 && *c == 0) return 0;
    APInt divisor;
    if (op->getOperand(0) == outer.getInductionVar() &&
        matchPattern(op->getOperand(1), m_ConstantInt(&divisor)) && divisor.isStrictlyPositive() &&
        divisor.getLimitedValue() % lanes == 0)
      return isa<arith::DivUIOp>(op) ? 0 : 1;
  }
  if (auto select = dyn_cast<arith::SelectOp>(op)) {
    auto condition = input(select.getCondition()), a = input(select.getTrueValue()), c = input(select.getFalseValue());
    if (condition && *condition == 0 && a && c && *a == *c) return a;
  }
  return {};
}
bool scalarOperation(Operation &op) {
  return op.getNumRegions() == 0 && isMemoryEffectFree(&op) &&
      (op.getName().getDialectNamespace() == "arith" || op.getName().getDialectNamespace() == "math") &&
      llvm::all_of(op.getResultTypes(), [](Type type) { return type.isIntOrIndexOrFloat(); });
}
}

FailureOr<weft::x86::OutputContraction> weft::x86::packetizeContractionOutputs(
    IRRewriter &b, scf::ForOp source, unsigned lanes) {
  auto outer = dyn_cast<scf::ForOp>(source->getParentOp());
  APInt lower, step;
  if (!outer || outer.getNumResults() ||
      !matchPattern(outer.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
      !matchPattern(outer.getStep(), m_ConstantInt(&step)) || !step.isOne() ||
      !weft::analysis::isReadEpoch(*source.getBody(), [](Operation *op) { return isa<memref::LoadOp>(op); }))
    return failure();
  SmallVector<Value> outputs;
  for (Operation &op : outer.getBody()->without_terminator()) {
    if (&op == source.getOperation()) continue;
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      Value root = storageRoot(store.getMemRef());
      if (!root.getDefiningOp<memref::AllocOp>() || store.getIndices().size() != 1 ||
          store.getIndices()[0] != outer.getInductionVar()) return failure();
      outputs.push_back(root);
    } else if (!scalarOperation(op)) return failure();
  }
  if (outputs.empty()) return failure();
  for (Value bound : {source.getLowerBound(), source.getUpperBound(), source.getStep()})
    if (coefficient(bound, outer, source, lanes) != std::optional<int64_t>(0)) return failure();
  for (Operation &op : source.getBody()->without_terminator()) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (load.getIndices().size() != 1 || llvm::is_contained(outputs, storageRoot(load.getMemRef()))) return failure();
    } else if (!scalarOperation(op)) return failure();
  }

  Location loc = source.getLoc();
  b.setInsertionPoint(outer);
  auto index = [&](int64_t n) -> Value { return arith::ConstantIndexOp::create(b, loc, n); };
  Value width = index(lanes);
  Value end = arith::SubIOp::create(b, loc, outer.getUpperBound(), arith::RemUIOp::create(b, loc, outer.getUpperBound(), width));
  auto packet = scf::ForOp::create(b, loc, outer.getLowerBound(), end, width);
  b.setInsertionPointToStart(packet.getBody());
  auto vectorType = [&](Type type) { return VectorType::get({int64_t(lanes)}, type); };
  auto broadcast = [&](Value value) -> Value {
    return isa<VectorType>(value.getType()) ? value : Value(vector::BroadcastOp::create(b, loc, vectorType(value.getType()), value));
  };
  SmallVector<int64_t> offsets;
  for (unsigned i = 0; i < lanes; ++i) offsets.push_back(i);
  Value indices = arith::AddIOp::create(b, loc, broadcast(packet.getInductionVar()),
      arith::ConstantOp::create(b, loc, vectorType(b.getIndexType()), DenseIntElementsAttr::get(vectorType(b.getIndexType()), offsets)));
  IRMapping mapping, scalar;
  scf::ForOp reduction;
  mapping.map(outer.getInductionVar(), indices);
  scalar.map(outer.getInductionVar(), packet.getInductionVar());
  auto scalarValue = [&](auto &&self, Value value) -> Value {
    if (scalar.contains(value)) return scalar.lookup(value);
    auto *op = value.getDefiningOp();
    if (!op || !outer->isProperAncestor(op)) return value;
    for (Value input : op->getOperands()) scalar.map(input, self(self, input));
    OpBuilder::InsertionGuard guard(b);
    if (reduction && op->getBlock() == outer.getBody()) b.setInsertionPoint(reduction);
    b.clone(*op, scalar);
    return scalar.lookup(value);
  };
  auto clone = [&](Operation &op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto stride = coefficient(load.getIndices()[0], outer, source, lanes);
      Value result;
      if (stride && (*stride == 0 || (*stride == 1 && !load.getType().isInteger(1)))) {
        Value base = scalarValue(scalarValue, load.getIndices()[0]);
        if (*stride == 0) result = memref::LoadOp::create(b, loc, load.getMemRef(), base);
        else result = vector::LoadOp::create(b, loc, vectorType(load.getType()), load.getMemRef(), base);
      } else {
        Value address = mapping.lookupOrDefault(load.getIndices()[0]);
        if (!isa<VectorType>(address.getType())) result = memref::LoadOp::create(b, loc, load.getMemRef(), address);
        else {
          Value mask = arith::ConstantOp::create(b, loc, vectorType(b.getI1Type()),
              DenseElementsAttr::get(vectorType(b.getI1Type()), b.getBoolAttr(true)));
          Value fill = arith::ConstantOp::create(b, loc, vectorType(load.getType()), b.getZeroAttr(vectorType(load.getType())));
          result = vector::GatherOp::create(b, loc, vectorType(load.getType()), load.getMemRef(), ValueRange{index(0)}, address, mask, fill);
        }
      }
      mapping.map(load.getResult(), result);
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      Value value = broadcast(mapping.lookupOrDefault(store.getValue()));
      if (store.getValue().getType().isInteger(1)) {
        Value active = arith::ConstantOp::create(b, loc, vectorType(b.getI1Type()),
            DenseElementsAttr::get(vectorType(b.getI1Type()), b.getBoolAttr(true)));
        vector::ScatterOp::create(b, loc, Type{}, store.getMemRef(), ValueRange{index(0)}, indices, active, value);
      } else vector::StoreOp::create(b, loc, value, store.getMemRef(), packet.getInductionVar());
      return;
    }
    SmallVector<Value> inputs;
    bool shaped = false;
    for (Value input : op.getOperands()) {
      Value value = mapping.lookupOrDefault(input);
      inputs.push_back(value); shaped |= isa<VectorType>(value.getType());
    }
    if (shaped) for (Value &value : inputs) value = broadcast(value);
    OperationState state(op.getLoc(), op.getName());
    state.addAttributes(op.getAttrs()); state.addOperands(inputs);
    for (Type type : op.getResultTypes()) state.addTypes(shaped ? Type(vectorType(type)) : type);
    mapping.map(op.getResults(), b.create(state)->getResults());
  };
  for (Operation &op : outer.getBody()->without_terminator()) {
    if (&op != source.getOperation()) { clone(op); continue; }
    SmallVector<Value> initial;
    for (Value value : source.getInitArgs()) initial.push_back(broadcast(mapping.lookupOrDefault(value)));
    reduction = scf::ForOp::create(b, loc, mapping.lookupOrDefault(source.getLowerBound()),
        mapping.lookupOrDefault(source.getUpperBound()), mapping.lookupOrDefault(source.getStep()), initial);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(reduction.getBody());
      mapping.map(source.getInductionVar(), reduction.getInductionVar());
      scalar.map(source.getInductionVar(), reduction.getInductionVar());
      mapping.map(source.getRegionIterArgs(), reduction.getRegionIterArgs());
      for (Operation &nested : source.getBody()->without_terminator()) clone(nested);
      SmallVector<Value> yielded;
      for (Value value : source.getBody()->getTerminator()->getOperands()) yielded.push_back(broadcast(mapping.lookupOrDefault(value)));
      scf::YieldOp::create(b, loc, yielded);
    }
    mapping.map(source.getResults(), reduction.getResults());
  }
  // Remove unused wide address constructions after choosing scalar/unit supply.
  // Keep all reads and effects; only dead pure expressions are erased.
  bool changed;
  do {
    changed = false;
    packet.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (!op->getNumRegions() && !op->hasTrait<OpTrait::IsTerminator>() && op->use_empty() && isMemoryEffectFree(op)) {
        b.eraseOp(op); changed = true;
      }
    });
  } while (changed);
  return OutputContraction{packet, reduction};
}
