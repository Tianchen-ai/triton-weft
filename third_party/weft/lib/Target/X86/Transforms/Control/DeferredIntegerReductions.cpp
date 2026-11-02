#include "Weft/Target/X86/X86Compiler.h"
#include "Analysis/Resources.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include <optional>

using namespace mlir;
namespace {
struct ReducedTerms {
  SmallVector<Value> inputs;
  Value coefficient;
  int64_t lanes;
};

// The physical reduction operand already includes the source validity fill.
// Thus an inactive term contributes zero without changing the separately
// carried source validity, including partial/empty groups and masked tails.
std::optional<ReducedTerms> reductionTerms(Value value, Block *body) {
  SmallVector<Value> scalars;
  if (auto elements = value.getDefiningOp<vector::FromElementsOp>())
    llvm::append_range(scalars, elements.getElements());
  else if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>();
           broadcast && cast<VectorType>(value.getType()).getNumElements() == 1)
    scalars.push_back(broadcast.getSource());
  else if (value.getType().isIntOrIndex()) scalars.push_back(value);
  else return std::nullopt;
  ReducedTerms result{{}, {}, 0};
  for (Value scalar : scalars) {
    auto reduction = scalar.getDefiningOp<vector::ReductionOp>();
    if (!reduction || reduction->getBlock() != body ||
        reduction.getKind() != vector::CombiningKind::ADD || reduction.getAcc())
      return std::nullopt;
    auto type = dyn_cast<VectorType>(reduction.getVector().getType());
    if (!type || type.isScalable() || type.getRank() != 1 ||
        type.getNumElements() < 2 || !isa<IntegerType>(type.getElementType()))
      return std::nullopt;
    if (result.lanes && result.lanes != type.getNumElements()) return std::nullopt;
    result.lanes = type.getNumElements();
    result.inputs.push_back(reduction.getVector());
  }
  return result.inputs.empty() ? std::nullopt : std::optional(result);
}

std::optional<ReducedTerms> weightedTerms(Value value, Block *body) {
  if (auto direct = reductionTerms(value, body)) return direct;
  auto product = value.getDefiningOp<arith::MulIOp>();
  if (!product || product.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return std::nullopt;
  for (unsigned side = 0; side < 2; ++side)
    if (auto terms = reductionTerms(product->getOperand(side), body)) {
      terms->coefficient = product->getOperand(1 - side);
      return terms;
    }
  return std::nullopt;
}

bool deferState(scf::ForOp source, unsigned position, IRRewriter &b) {
  Value carry = source.getRegionIterArgs()[position];
  auto type = carry.getType();
  auto vector = dyn_cast<VectorType>(type);
  auto integer = dyn_cast<IntegerType>(vector ? vector.getElementType() : type);
  if (!integer || (vector && (vector.isScalable() || vector.getRank() != 1)) ||
      !carry.hasOneUse()) return false;
  auto update = dyn_cast<arith::AddIOp>(*carry.getUsers().begin());
  if (!update || update->getBlock() != source.getBody() || !update->hasOneUse() ||
      update.getOverflowFlags() != arith::IntegerOverflowFlags::none) return false;
  auto yield = cast<scf::YieldOp>(source.getBody()->getTerminator());
  if (yield.getOperand(position) != update.getResult()) return false;
  Value term = update.getLhs() == carry ? update.getRhs() : update.getLhs();
  auto terms = weightedTerms(term, source.getBody());
  if (!terms) return false;
  int64_t outputs = vector ? vector.getNumElements() : 1;
  if (int64_t(terms->inputs.size()) != outputs) return false;
  auto model = weft::x86::ResourceModel::get(source);
  if (outputs * terms->lanes * integer.getWidth() >
      int64_t(model.vectorBits) * std::max(1u, model.vectorRegisters / 2)) return false;

  auto expandedType = VectorType::get({outputs * terms->lanes}, integer);
  auto partType = VectorType::get({terms->lanes}, integer);
  Location loc = source.getLoc();
  b.setInsertionPoint(source);
  Value initial = arith::ConstantOp::create(b, loc, expandedType,
      DenseElementsAttr::get(expandedType, b.getIntegerAttr(integer, 0)));
  for (int64_t output = 0; output < outputs; ++output) {
    Value seed = source.getInitArgs()[position];
    if (vector) seed = vector::ExtractOp::create(b, loc, seed, ArrayRef<int64_t>{output});
    // One lane receives the author seed. Splatting it would multiply it by
    // the physical width and would also change a zero-trip loop's result.
    initial = vector::InsertOp::create(b, loc, seed, initial,
        ArrayRef<int64_t>{output * terms->lanes});
  }
  SmallVector<Value> seeds(source.getInitArgs());
  seeds[position] = initial;
  auto loop = scf::ForOp::create(b, loc, source.getLowerBound(),
      source.getUpperBound(), source.getStep(), seeds);
  loop->setAttrs(source->getAttrs());
  loop->setAttr("weft.x86.deferred_integer_reduction", b.getI64IntegerAttr(terms->lanes));
  b.setInsertionPointToStart(loop.getBody());
  IRMapping mapping;
  mapping.map(source.getInductionVar(), loop.getInductionVar());
  mapping.map(source.getRegionIterArgs(), loop.getRegionIterArgs());
  for (Operation &op : source.getBody()->without_terminator())
    if (&op != update.getOperation()) b.clone(op, mapping);
  Value contribution = arith::ConstantOp::create(b, loc, expandedType,
      DenseElementsAttr::get(expandedType, b.getIntegerAttr(integer, 0)));
  for (int64_t output = 0; output < outputs; ++output) {
    Value partial = mapping.lookupOrDefault(terms->inputs[output]);
    if (terms->coefficient) {
      Value factor = mapping.lookupOrDefault(terms->coefficient);
      if (isa<VectorType>(factor.getType()))
        factor = vector::ExtractOp::create(b, loc, factor, ArrayRef<int64_t>{output});
      factor = vector::BroadcastOp::create(b, loc, partType, factor);
      partial = arith::MulIOp::create(b, loc, partial, factor);
    }
    contribution = vector::InsertStridedSliceOp::create(b, loc, partial,
        contribution, ArrayRef<int64_t>{output * terms->lanes}, ArrayRef<int64_t>{1});
  }
  Value next = arith::AddIOp::create(b, loc, loop.getRegionIterArgs()[position], contribution);
  SmallVector<Value> yielded;
  for (auto [i, value] : llvm::enumerate(yield.getOperands()))
    yielded.push_back(i == position ? next : mapping.lookupOrDefault(value));
  if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back()))
    b.eraseOp(&loop.getBody()->back());
  b.setInsertionPointToEnd(loop.getBody());
  scf::YieldOp::create(b, loc, yielded);

  b.setInsertionPointAfter(loop);
  SmallVector<Value> sums;
  for (int64_t output = 0; output < outputs; ++output) {
    Value partial = vector::ExtractStridedSliceOp::create(b, loc,
        loop.getResult(position), ArrayRef<int64_t>{output * terms->lanes},
        ArrayRef<int64_t>{terms->lanes}, ArrayRef<int64_t>{1});
    sums.push_back(vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, partial));
  }
  Value result = vector ? Value(vector::FromElementsOp::create(b, loc, type, sums)) : sums.front();
  SmallVector<Value> results(loop.getResults());
  results[position] = result;
  auto function = source->getParentOfType<func::FuncOp>();
  if (function) {
    auto count = function->getAttrOfType<IntegerAttr>("weft.x86.deferred_integer_states");
    function->setAttr("weft.x86.deferred_integer_states",
        b.getI64IntegerAttr((count ? count.getInt() : 0) + 1));
  }
  b.replaceOp(source, results);
  return true;
}

class DeferredIntegerReductions
    : public PassWrapper<DeferredIntegerReductions, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeferredIntegerReductions)
  StringRef getArgument() const override { return "weft-x86-defer-integer-reductions"; }
  StringRef getDescription() const override {
    return "Retain modular integer state in partial vectors until its source consumer";
  }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    IRRewriter b(&getContext());
    for (auto loop : loops)
      for (unsigned i = 0; i < loop.getNumRegionIterArgs(); ++i)
        if (deferState(loop, i, b)) break;
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86DeferredIntegerReductionPass() {
  return std::make_unique<DeferredIntegerReductions>();
}
