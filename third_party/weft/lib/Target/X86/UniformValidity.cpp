#include "Weft/Target/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;
namespace {
bool same(Value a, Value c) {
  APInt x, y;
  return a == c || (matchPattern(a, m_ConstantInt(&x)) && matchPattern(c, m_ConstantInt(&y)) && x == y);
}
bool integer(Value value, uint64_t expected) {
  APInt number;
  return matchPattern(value, m_ConstantInt(&number)) && number == expected;
}
std::optional<APInt> splatInteger(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant) return {};
  if (auto scalar = dyn_cast<IntegerAttr>(constant.getValue())) return scalar.getValue();
  if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue()); dense && dense.isSplat())
    return dense.getSplatValue<APInt>();
  return {};
}

void foldUniformBitsets(ModuleOp module, IRRewriter &b) {
  SmallVector<arith::TruncIOp> truncations;
  module.walk([&](arith::TruncIOp op) {
    if (getElementTypeOrSelf(op.getType()).isInteger(1)) truncations.push_back(op);
  });
  for (auto narrow : truncations) {
    auto shift = narrow.getIn().getDefiningOp<arith::ShRUIOp>();
    if (!shift) continue;
    auto bits = splatInteger(shift.getLhs());
    if (!bits || (!bits->isAllOnes() && !bits->isZero())) continue;
    auto bound = splatInteger(shift.getRhs());
    if (!bound)
      if (auto mask = shift.getRhs().getDefiningOp<arith::AndIOp>()) {
        bound = splatInteger(mask.getLhs());
        if (!bound) bound = splatInteger(mask.getRhs());
      }
    // A masked index is an unsigned bound, even when its original signed
    // carrier can be negative. Keep shifts with no proven in-range amount.
    if (!bound || bound->uge(bits->getBitWidth())) continue;
    b.setInsertionPoint(narrow);
    TypedAttr result = b.getBoolAttr(bits->isAllOnes());
    if (auto vector = dyn_cast<VectorType>(narrow.getType()))
      result = DenseElementsAttr::get(vector, result);
    b.replaceOpWithNewOp<arith::ConstantOp>(narrow, narrow.getType(), result);
  }
}
Value uniformPredicate(Value value, memref::AllocOp alloc, DominanceInfo &dominance, IRRewriter &b) {
  if (value.getType().isInteger(1) && dominance.dominates(value, alloc)) return value;
  if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>())
    return uniformPredicate(broadcast.getSource(), alloc, dominance, b);
  if (auto widen = value.getDefiningOp<arith::ExtUIOp>())
    return uniformPredicate(widen.getIn(), alloc, dominance, b);
  if (auto narrow = value.getDefiningOp<arith::TruncIOp>())
    return uniformPredicate(narrow.getIn(), alloc, dominance, b);
  if (auto reduction = value.getDefiningOp<vector::ReductionOp>()) {
    auto kind = reduction.getKind();
    // AND/OR of a nonempty uniform predicate is the same predicate. Closed
    // contraction schedules can leave this form after their carries simplify;
    // retain the complete-writer and dominance proof for the surrounding array.
    if (value.getType().isInteger(1) && !reduction.getAcc() &&
        (kind == vector::CombiningKind::AND || kind == vector::CombiningKind::OR))
      return uniformPredicate(reduction.getVector(), alloc, dominance, b);
  }
  Attribute attribute;
  if (matchPattern(value, m_Constant(&attribute))) {
    APInt bits;
    if (auto scalar = dyn_cast<IntegerAttr>(attribute)) bits = scalar.getValue();
    else if (auto dense = dyn_cast<DenseIntElementsAttr>(attribute); dense && dense.isSplat())
      bits = dense.getSplatValue<APInt>();
    else return {};
    if (bits != 0 && bits != 1) return {};
    b.setInsertionPoint(alloc);
    return arith::ConstantIntOp::create(b, alloc.getLoc(), bits == 1, 1);
  }
  return {};
}
Operation *inBlock(Operation *operation, Block *block) {
  while (operation && operation->getBlock() != block) operation = operation->getParentOp();
  return operation;
}
struct Writer { Operation *store; scf::ForOp loop; Value value; unsigned lanes; };

bool uniformLanes(Value value, const llvm::DenseSet<Value> &carries,
                  llvm::DenseMap<Value, bool> &known) {
  auto type = dyn_cast<VectorType>(value.getType());
  if (!type) return true;
  if (type.isScalable()) return false;
  if (type.getNumElements() == 1 || carries.contains(value)) return true;
  auto found = known.find(value);
  if (found != known.end()) return found->second;
  auto test = [&](Value input) { return uniformLanes(input, carries, known); };
  bool result = false;
  Attribute constant;
  if (matchPattern(value, m_Constant(&constant))) {
    if (auto elements = dyn_cast<DenseElementsAttr>(constant)) result = elements.isSplat();
  } else if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>()) {
    result = test(broadcast.getSource());
  } else if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
    int64_t width = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
    bool first = false, second = false;
    for (int64_t lane : shuffle.getMask()) {
      if (lane < 0) return false;
      first |= lane < width; second |= lane >= width;
    }
    if (!second) result = test(shuffle.getV1());
    else if (!first) result = test(shuffle.getV2());
    else result = shuffle.getV1() == shuffle.getV2() && test(shuffle.getV1());
  } else if (auto choose = value.getDefiningOp<scf::IfOp>()) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    auto yes = cast<scf::YieldOp>(choose.thenBlock()->getTerminator());
    auto no = cast<scf::YieldOp>(choose.elseBlock()->getTerminator());
    result = test(yes.getOperand(index)) && test(no.getOperand(index));
  } else if (Operation *op = value.getDefiningOp()) {
    if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::CmpIOp,
            arith::SelectOp, arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp,
            vector::ShapeCastOp, vector::ExtractStridedSliceOp, vector::ExtractOp>(op))
      result = llvm::all_of(op->getOperands(), test);
  }
  known[value] = result;
  return result;
}

void scalarizeCarries(scf::ForOp loop, IRRewriter &b) {
  llvm::DenseSet<Value> candidates;
  llvm::DenseMap<Value, bool> known;
  for (auto [init, arg] : llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs())) {
    auto type = dyn_cast<VectorType>(init.getType());
    if (type && type.getRank() == 1 && type.getElementType().isInteger(1) &&
        !type.isScalable() && uniformLanes(init, {}, known)) candidates.insert(arg);
  }
  if (candidates.empty()) return;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  // Mutual carries are an induction hypothesis, not proof by circular use:
  // each candidate has a uniform initial value and a uniform update under
  // that hypothesis. Remove dependencies on any rejected carry to a fixed point.
  bool changed;
  do {
    changed = false; known.clear();
    for (auto [arg, next] : llvm::zip(loop.getRegionIterArgs(), yield.getOperands())) {
      if (candidates.contains(arg) && !uniformLanes(next, candidates, known)) {
        candidates.erase(arg); changed = true;
      }
    }
  } while (changed);
  if (candidates.empty()) return;
  b.setInsertionPoint(loop);
  SmallVector<Value> initial;
  for (auto [init, arg] : llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs()))
    initial.push_back(candidates.contains(arg) ?
        Value(vector::ExtractOp::create(b, loop.getLoc(), init, ArrayRef<int64_t>{0})) : init);
  auto replacement = scf::ForOp::create(b, loop.getLoc(), loop.getLowerBound(),
      loop.getUpperBound(), loop.getStep(), initial);
  replacement->setAttrs(loop->getAttrs());
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), replacement.getInductionVar());
  b.setInsertionPointToStart(replacement.getBody());
  for (auto [old, arg] : llvm::zip(loop.getRegionIterArgs(), replacement.getRegionIterArgs())) {
    Value mapped = arg;
    if (candidates.contains(old))
      mapped = vector::BroadcastOp::create(b, loop.getLoc(), old.getType(), arg);
    mapping.map(old, mapped);
  }
  for (Operation &op : loop.getBody()->without_terminator()) b.clone(op, mapping);
  SmallVector<Value> next;
  for (auto [arg, value] : llvm::zip(loop.getRegionIterArgs(), yield.getOperands())) {
    Value mapped = mapping.lookupOrDefault(value);
    if (candidates.contains(arg))
      mapped = vector::ExtractOp::create(b, yield.getLoc(), mapped, ArrayRef<int64_t>{0});
    next.push_back(mapped);
  }
  scf::YieldOp::create(b, yield.getLoc(), next);
  b.setInsertionPointAfter(replacement);
  SmallVector<Value> outputs;
  for (auto [old, value] : llvm::zip(loop.getRegionIterArgs(), replacement.getResults()))
    outputs.push_back(candidates.contains(old) ?
        Value(vector::BroadcastOp::create(b, loop.getLoc(), old.getType(), value)) : value);
  auto function = loop->getParentOfType<func::FuncOp>();
  auto prior = function->getAttrOfType<IntegerAttr>("weft.x86.uniform_predicate_carries");
  function->setAttr("weft.x86.uniform_predicate_carries",
      b.getI64IntegerAttr((prior ? prior.getInt() : 0) + candidates.size()));
  b.replaceOp(loop, outputs);
}

bool fold(memref::AllocOp alloc, IRRewriter &b) {
  if (alloc.getType().getRank() != 1 || !alloc.getType().getElementType().isInteger(8)) return false;
  SmallVector<Writer> writers;
  SmallVector<Operation *> reads;
  scf::ForOp rectangle;
  int64_t rectangleElements = 0;
  for (Operation *user : alloc.getResult().getUsers()) {
    Value stored; ValueRange indices; unsigned lanes = 1;
    if (auto store = dyn_cast<memref::StoreOp>(user)) { stored = store.getValue(); indices = store.getIndices(); }
    else if (auto store = dyn_cast<vector::StoreOp>(user)) {
      stored = store.getValueToStore(); indices = store.getIndices(); lanes = store.getVectorType().getNumElements();
    } else if (isa<memref::LoadOp, vector::LoadOp, vector::GatherOp>(user)) { reads.push_back(user); continue; }
    else return false; // ABI calls, escaping descriptors and other users retain their array.
    auto loop = dyn_cast_or_null<scf::ForOp>(user->getParentOp());
    if (!loop || indices.size() != 1 || !integer(loop.getStep(), lanes)) return false;
    if (indices[0] != loop.getInductionVar()) {
      auto outer = dyn_cast<scf::ForOp>(loop->getParentOp());
      auto address = indices[0].getDefiningOp<arith::AddIOp>();
      APInt width, height;
      if (rectangle || !outer || !address || outer->getBlock() != alloc->getBlock() ||
          !loop.getInitArgs().empty() || !outer.getInitArgs().empty() ||
          !integer(loop.getLowerBound(), 0) || !integer(outer.getLowerBound(), 0) ||
          !integer(outer.getStep(), 1) ||
          !matchPattern(loop.getUpperBound(), m_ConstantInt(&width)) || width.getSExtValue() <= 0 ||
          width.urem(lanes) != 0 ||
          !matchPattern(outer.getUpperBound(), m_ConstantInt(&height)) || height.getSExtValue() <= 0 ||
          height.getSExtValue() > std::numeric_limits<int64_t>::max() / width.getSExtValue()) return false;
      Value rowBase;
      if (address.getLhs() == loop.getInductionVar()) rowBase = address.getRhs();
      else if (address.getRhs() == loop.getInductionVar()) rowBase = address.getLhs();
      auto product = rowBase ? rowBase.getDefiningOp<arith::MulIOp>() : arith::MulIOp();
      if (!product || !((product.getLhs() == outer.getInductionVar() && same(product.getRhs(), loop.getUpperBound())) ||
                        (product.getRhs() == outer.getInductionVar() && same(product.getLhs(), loop.getUpperBound()))))
        return false;
      rectangle = outer;
      rectangleElements = width.getSExtValue() * height.getSExtValue();
    }
    writers.push_back({user, loop, stored, lanes});
  }
  if (writers.empty() || writers.size() > 2 || reads.empty()) return false;
  Value count;
  if (alloc.getType().isDynamicDim(0)) {
    count = alloc.getDynamicSizes().front();
  } else {
    b.setInsertionPoint(alloc);
    count = arith::ConstantIndexOp::create(b, alloc.getLoc(), alloc.getType().getDimSize(0));
  }
  Operation *initialized;
  bool sequential = llvm::all_of(writers, [&](const Writer &writer) {
    return writer.loop->getBlock() == alloc->getBlock();
  });
  if (rectangle) {
    if (writers.size() != 1 || !integer(count, rectangleElements)) return false;
    initialized = rectangle;
  } else if (sequential) {
    llvm::sort(writers, [](const auto &a, const auto &c) { return a.loop->isBeforeInBlock(c.loop); });
    if (!integer(writers.front().loop.getLowerBound(), 0) || !same(writers.back().loop.getUpperBound(), count)) return false;
    if (writers.size() == 2 && (!same(writers[0].loop.getUpperBound(), writers[1].loop.getLowerBound()) ||
                              writers[1].lanes != 1)) return false;
    initialized = writers.back().loop;
  } else {
    // Two mutually exclusive complete initializers establish the same bytes
    // after the choice. Neither branch alone establishes unconditional data.
    if (writers.size() != 2) return false;
    auto choice = dyn_cast_or_null<scf::IfOp>(writers[0].loop->getParentOp());
    if (!choice || choice->getBlock() != alloc->getBlock() ||
        writers[1].loop->getParentOp() != choice.getOperation() ||
        writers[0].loop->getBlock() == writers[1].loop->getBlock()) return false;
    for (auto &writer : writers)
      if (!integer(writer.loop.getLowerBound(), 0) || !same(writer.loop.getUpperBound(), count)) return false;
    initialized = choice;
  }
  for (auto &writer : writers) {
    if (writer.lanes == 1) continue;
    APInt bound;
    Value end = writer.loop.getUpperBound();
    bool complete = matchPattern(end, m_ConstantInt(&bound)) && !bound.isNegative() && bound.urem(writer.lanes) == 0;
    if (auto sub = end.getDefiningOp<arith::SubIOp>())
      if (auto rem = sub.getRhs().getDefiningOp<arith::RemUIOp>())
        complete |= same(sub.getLhs(), count) && same(rem.getLhs(), count) && integer(rem.getRhs(), writer.lanes);
    if (!complete) return false;
  }
  for (Operation *read : reads) {
    Operation *at = inBlock(read, alloc->getBlock());
    if (!at || !initialized->isBeforeInBlock(at)) return false;
  }
  DominanceInfo dominance(alloc->getParentOfType<func::FuncOp>());
  Value predicate;
  for (const auto &writer : writers) {
    Value value = uniformPredicate(writer.value, alloc, dominance, b);
    if (!value || (predicate && !same(predicate, value))) return false;
    predicate = value;
  }
  b.setInsertionPoint(alloc);
  Value byte = arith::ExtUIOp::create(b, alloc.getLoc(), b.getI8Type(), predicate);
  for (Operation *read : reads) {
    b.setInsertionPoint(read); Value replacement = byte;
    if (auto type = dyn_cast<VectorType>(read->getResult(0).getType()))
      replacement = vector::BroadcastOp::create(b, read->getLoc(), type, byte);
    if (auto gather = dyn_cast<vector::GatherOp>(read))
      replacement = arith::SelectOp::create(b, read->getLoc(), gather.getMask(), replacement, gather.getPassThru());
    b.replaceOp(read, replacement);
  }
  for (auto writer : writers) b.eraseOp(writer.store);
  auto function = alloc->getParentOfType<func::FuncOp>();
  auto old = function->getAttrOfType<IntegerAttr>("weft.x86.uniform_validity_arrays");
  function->setAttr("weft.x86.uniform_validity_arrays", b.getI64IntegerAttr((old ? old.getInt() : 0) + 1));
  b.eraseOp(alloc);
  return true;
}
class UniformValidity : public PassWrapper<UniformValidity, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UniformValidity)
  StringRef getArgument() const override { return "weft-x86-fold-uniform-validity"; }
  StringRef getDescription() const override { return "Use scalar predicates for proven uniform validity arrays and loop carries"; }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    foldUniformBitsets(getOperation(), b);
    SmallVector<memref::AllocOp> arrays;
    getOperation().walk([&](memref::AllocOp alloc) {
      // MemRef canonicalization can rebuild a constant-size allocation without
      // the producer's marker. The complete-writer and non-escape proof in
      // fold() establishes the byte contents directly, including numeric 0/1
      // arrays; a lost annotation must not force materialized predicates.
      if (alloc.getType().getRank() == 1 && alloc.getType().getElementType().isInteger(8))
        arrays.push_back(alloc);
    });
    for (auto alloc : arrays) fold(alloc, b);
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    for (auto loop : loops) scalarizeCarries(loop, b);
  }
};
} // namespace
std::unique_ptr<Pass> weft::createX86UniformValidityPass() { return std::make_unique<UniformValidity>(); }
