#include "Weft/Target/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>

using namespace mlir;
namespace {

std::optional<bool> fullComparison(Value value, Value iv, Value upper) {
  auto compare = value.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getLhs() != iv || compare.getRhs() != upper) return {};
  switch (compare.getPredicate()) {
  case arith::CmpIPredicate::ult: case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::slt: case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ne: return true;
  case arith::CmpIPredicate::ugt: case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::sgt: case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::eq: return false;
  }
  return {};
}

bool atLeast(Value value, int64_t minimum, Value iv, Value upper,
             int64_t step, unsigned depth = 0) {
  if (depth > 16) return false;
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return !constant.isNegative() && constant.getActiveBits() < 63 && constant.getZExtValue() >= uint64_t(minimum);
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    if (sub.getLhs() == upper && sub.getRhs() == iv) return step >= minimum;
    APInt offset;
    if (matchPattern(sub.getRhs(), m_ConstantInt(&offset)) && !offset.isNegative() &&
        offset.getActiveBits() < 63 && minimum <= INT64_MAX - int64_t(offset.getZExtValue()))
      return atLeast(sub.getLhs(), minimum + offset.getZExtValue(), iv, upper, step, depth + 1);
    if (sub.getLhs() == upper) {
      auto add = sub.getRhs().getDefiningOp<arith::AddIOp>();
      if (add) for (unsigned side = 0; side < 2; ++side)
        if (add->getOperand(side) == iv &&
            matchPattern(add->getOperand(1 - side), m_ConstantInt(&offset)) &&
            !offset.isNegative() && offset.getActiveBits() < 63 &&
            minimum <= INT64_MAX - int64_t(offset.getZExtValue()))
          return step >= minimum + int64_t(offset.getZExtValue());
    }
    return false;
  }
  auto recurse = [&](Value v) { return atLeast(v, minimum, iv, upper, step, depth + 1); };
  if (auto min = value.getDefiningOp<arith::MinUIOp>()) return recurse(min.getLhs()) && recurse(min.getRhs());
  if (auto div = value.getDefiningOp<arith::DivUIOp>()) {
    APInt divisor;
    if (!matchPattern(div.getRhs(), m_ConstantInt(&divisor)) || divisor.isZero() ||
        divisor.isNegative() || divisor.getActiveBits() >= 63) return false;
    int64_t factor = divisor.getZExtValue();
    if (minimum > INT64_MAX / factor) return false;
    return atLeast(div.getLhs(), minimum * factor, iv, upper, step, depth + 1);
  }
  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    APInt condition;
    if (matchPattern(select.getCondition(), m_ConstantInt(&condition)))
      return recurse(condition.isZero() ? select.getFalseValue() : select.getTrueValue());
    if (auto known = fullComparison(select.getCondition(), iv, upper))
      return recurse(*known ? select.getTrueValue() : select.getFalseValue());
    return recurse(select.getTrueValue()) && recurse(select.getFalseValue());
  }
  return false;
}

std::optional<bool> completeTileComparison(arith::CmpIOp compare, Value iv,
                                         Value upper, int64_t step) {
  if (auto known = fullComparison(compare.getResult(), iv, upper)) return known;
  Value value = compare.getLhs();
  APInt limit;
  auto predicate = compare.getPredicate();
  if (!matchPattern(compare.getRhs(), m_ConstantInt(&limit))) {
    if (!matchPattern(compare.getLhs(), m_ConstantInt(&limit))) return {};
    value = compare.getRhs();
    switch (predicate) {
    case arith::CmpIPredicate::ult: predicate = arith::CmpIPredicate::ugt; break;
    case arith::CmpIPredicate::ule: predicate = arith::CmpIPredicate::uge; break;
    case arith::CmpIPredicate::ugt: predicate = arith::CmpIPredicate::ult; break;
    case arith::CmpIPredicate::uge: predicate = arith::CmpIPredicate::ule; break;
    default: return {};
    }
  }
  if (limit.isNegative() || limit.getActiveBits() >= 63) return {};
  int64_t bound = limit.getZExtValue();
  // Encoded record validity tests remaining >= record_extent directly, even
  // after the element-domain min has folded. Prove these predicates from the
  // same complete-loop bound; the cloned remainder keeps its original masks.
  switch (predicate) {
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::ult:
    if (atLeast(value, bound, iv, upper, step))
      return predicate == arith::CmpIPredicate::uge;
    break;
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::ule:
    if (atLeast(value, bound + 1, iv, upper, step))
      return predicate == arith::CmpIPredicate::ugt;
    break;
  default: break;
  }
  return {};
}

Value completeMinimum(arith::MinUIOp min, Value iv, Value upper, int64_t step) {
  for (unsigned i = 0; i < 2; ++i) {
    APInt limit;
    Value bound = min->getOperand(i);
    if (matchPattern(bound, m_ConstantInt(&limit)) && !limit.isNegative() &&
        limit.getActiveBits() < 63 && limit.getZExtValue() <= uint64_t(step) &&
        atLeast(min->getOperand(1 - i), limit.getZExtValue(), iv, upper, step))
      return bound;
  }
  return {};
}

struct FoldMaskReduction : OpRewritePattern<vector::ReductionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ReductionOp op, PatternRewriter &b) const override {
    if (op.getAcc() || !op.getType().isInteger(1) || (op.getKind() != vector::CombiningKind::AND &&
        op.getKind() != vector::CombiningKind::OR)) return failure();
    DenseIntElementsAttr mask;
    if (!matchPattern(op.getVector(), m_Constant(&mask))) return failure();
    bool result = op.getKind() == vector::CombiningKind::AND;
    for (APInt bit : mask.getValues<APInt>())
      result = op.getKind() == vector::CombiningKind::AND ? result && !bit.isZero() : result || !bit.isZero();
    b.replaceOpWithNewOp<arith::ConstantIntOp>(op, result, 1);
    return success();
  }
};

std::optional<int64_t> tileStep(scf::ForOp loop) {
  APInt lower, stride;
  if (!loop.getInductionVar().getType().isIndex() ||
      !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
      !matchPattern(loop.getStep(), m_ConstantInt(&stride)) || stride.isNegative() ||
      stride.getActiveBits() >= 63 || stride.getZExtValue() <= 1) return {};
  return stride.getZExtValue();
}

void foldCompleteBounds(scf::ForOp loop, int64_t step, IRRewriter &b) {
  Value iv = loop.getInductionVar(), upper = loop.getUpperBound();
  bool changed;
  do {
    changed = false;
    loop.getBody()->walk([&](arith::MinUIOp min) {
      if (Value bound = completeMinimum(min, iv, upper, step)) {
        b.replaceOp(min, bound);
        changed = true;
      }
    });
  } while (changed);
  SmallVector<arith::CmpIOp> comparisons;
  loop.getBody()->walk([&](arith::CmpIOp op) { comparisons.push_back(op); });
  for (auto compare : comparisons) {
    auto known = completeTileComparison(compare, iv, upper, step);
    if (!known) continue;
    b.setInsertionPoint(compare);
    b.replaceOpWithNewOp<arith::ConstantIntOp>(compare, *known, 1);
  }
}

class SeparateFullTiles
    : public PassWrapper<SeparateFullTiles, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SeparateFullTiles)
  StringRef getArgument() const override { return "weft-x86-separate-full-tiles"; }
  StringRef getDescription() const override {
    return "Separate complete tile iterations from the masked remainder without changing traversal or effects";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) { loops.push_back(loop); });
    IRRewriter b(&getContext());
    RewritePatternSet simplification(&getContext());
    simplification.add<FoldMaskReduction>(&getContext());
    scf::IfOp::getCanonicalizationPatterns(simplification, &getContext());
    scf::ForOp::getCanonicalizationPatterns(simplification, &getContext());
    vector::ShuffleOp::getCanonicalizationPatterns(simplification, &getContext());
    FrozenRewritePatternSet patterns(std::move(simplification));
    // Bound growth per function relative to the program already present.
    // A fixed module-wide allowance makes independently useful full-tile
    // paths compete when authors compose quantization and contraction.
    // Keep an absolute cap as well as the small-function policy allowance.
    llvm::DenseMap<Operation *, int64_t> budgets, duplicated, versions;
    getOperation().walk([&](func::FuncOp function) {
      if (function.isExternal()) return;
      int64_t original = 0;
      function.walk([&](Operation *) { ++original; });
      budgets[function] = std::clamp<int64_t>(original, 6000, 24000);
    });
    // Independent tiled axes can share one complete-domain version. Peeling
    // each inner axis first spends the growth budget before a parent mask can
    // disappear, leaving that mask live across all inner products. Version the
    // original nest once when its bounds are already available at the entry.
    // The alternate branch keeps the exact original traversal and carries.
    for (auto root : llvm::reverse(loops)) {
      if (root->getParentOfType<scf::ForOp>() || !tileStep(root)) continue;
      SmallVector<scf::ForOp> axes;
      int64_t operations = 0;
      root.walk([&](Operation *) { ++operations; });
      if (operations > 24000) continue;
      root.walk<WalkOrder::PreOrder>([&](scf::ForOp axis) {
        auto step = tileStep(axis);
        if (!step || !root.isDefinedOutsideOfLoop(axis.getUpperBound())) return;
        bool useful = false;
        axis.getBody()->walk([&](arith::MinUIOp min) {
          useful |= bool(completeMinimum(min, axis.getInductionVar(), axis.getUpperBound(), *step));
        });
        if (useful) axes.push_back(axis);
      });
      if (axes.size() < 2 || axes.front() != root) continue;
      auto function = root->getParentOfType<func::FuncOp>();
      if (!function) continue;
      b.setInsertionPoint(root);
      auto full = cast<scf::ForOp>(b.clone(*root.getOperation()));
      SmallVector<scf::ForOp> fullAxes;
      full.walk<WalkOrder::PreOrder>([&](scf::ForOp axis) {
        if (tileStep(axis) && full.isDefinedOutsideOfLoop(axis.getUpperBound()))
          fullAxes.push_back(axis);
      });
      // Only bounds included in the entry condition justify mask removal.
      for (auto axis : fullAxes) {
        bool guarded = llvm::any_of(axes, [&](scf::ForOp original) {
          return original.getUpperBound() == axis.getUpperBound() &&
                 tileStep(original) == tileStep(axis);
        });
        if (guarded) foldCompleteBounds(axis, *tileStep(axis), b);
      }
      SmallVector<Operation *> proposed;
      full.getBody()->walk([&](Operation *op) { proposed.push_back(op); });
      GreedyRewriteConfig config;
      config.setScope(&full.getRegion()).setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
      if (failed(applyOpPatternsGreedily(proposed, patterns, config))) {
        signalPassFailure(); return;
      }
      int64_t added = 0;
      full.getBody()->walk([&](Operation *) { ++added; });
      if (added > budgets.lookup(function) - duplicated.lookup(function)) {
        b.eraseOp(full); continue;
      }
      b.setInsertionPoint(full);
      Value condition = arith::ConstantIntOp::create(b, root.getLoc(), true, 1);
      Value zero = arith::ConstantIndexOp::create(b, root.getLoc(), 0);
      for (auto axis : axes) {
        Value stride = arith::ConstantIndexOp::create(b, root.getLoc(), *tileStep(axis));
        Value nonnegative = arith::CmpIOp::create(b, root.getLoc(), arith::CmpIPredicate::sge,
                                                axis.getUpperBound(), zero);
        Value remainder = arith::RemUIOp::create(b, root.getLoc(), axis.getUpperBound(), stride);
        Value divisible = arith::CmpIOp::create(b, root.getLoc(), arith::CmpIPredicate::eq,
                                              remainder, zero);
        condition = arith::AndIOp::create(b, root.getLoc(), condition,
                                        arith::AndIOp::create(b, root.getLoc(), nonnegative, divisible));
      }
      auto branch = scf::IfOp::create(b, root.getLoc(), root.getResultTypes(), condition, true);
      root->replaceAllUsesWith(branch.getResults());
      for (auto [body, loop] : {std::pair{&branch.getThenRegion().front(), full},
                                std::pair{&branch.getElseRegion().front(), root}}) {
        if (body->mightHaveTerminator()) b.eraseOp(body->getTerminator());
        loop->moveBefore(body, body->end());
        b.setInsertionPointToEnd(body);
        scf::YieldOp::create(b, root.getLoc(), loop.getResults());
      }
      duplicated[function] += added;
      ++versions[function];
    }
    for (auto loop : loops) {
      // The original alternate nest still benefits from inner complete tiles
      // when another axis has a tail. Spend only the remaining shared budget;
      // the complete-domain clone is not present in this original loop list.
      // A scalar carry can consume a large SIMD body. Its result type does not
      // establish that the body has no vector masks to eliminate; use the
      // actual complete-tile proofs and bounded body size below instead.
      APInt lower, stride;
      if (!loop.getInductionVar().getType().isIndex() ||
          !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
          !matchPattern(loop.getStep(), m_ConstantInt(&stride)) || stride.isNegative() ||
          stride.getActiveBits() >= 63 || stride.getZExtValue() <= 1) continue;
      Value iv = loop.getInductionVar(), upper = loop.getUpperBound();
      int64_t step = stride.getZExtValue(), operations = 0;
      bool hasMinimum = false;
      loop.getBody()->walk([&](Operation *op) {
        ++operations;
        auto min = dyn_cast<arith::MinUIOp>(op);
        if (!min) return;
        hasMinimum |= bool(completeMinimum(min, iv, upper, step));
      });
      // Parent row/column validity is consumed inside reduction loops too.
      // Visit original loops inside-out, but bound total cloned work so nested
      // full/tail versions cannot grow without limit. Newly cloned loops are
      // not reconsidered by this pass.
      auto function = loop->getParentOfType<func::FuncOp>();
      if (!function) continue;
      int64_t &duplicatedOperations = duplicated[function];
      int64_t budget = budgets.lookup(function);
      if (!hasMinimum || operations > 24000 || duplicatedOperations >= budget) continue;
      b.setInsertionPoint(loop);
      auto full = cast<scf::ForOp>(b.clone(*loop.getOperation()));
      Value fullIV = full.getInductionVar();
      // A View clips its extent by both the Level domain and the remaining
      // backing storage. Replacing the Level minimum exposes a constant in
      // those dependent clips. Fold them while the original complete-loop
      // bound is still available; a later generic canonicalizer cannot prove
      // upper - iv >= step from the rounded-down loop bound by itself.
      bool changed;
      do {
        changed = false;
        full.getBody()->walk([&](arith::MinUIOp min) {
          if (Value bound = completeMinimum(min, fullIV, upper, step)) {
            b.replaceOp(min, bound);
            changed = true;
          }
        });
      } while (changed);
      SmallVector<arith::CmpIOp> comparisons;
      full.getBody()->walk([&](arith::CmpIOp compare) { comparisons.push_back(compare); });
      for (auto compare : comparisons) {
        auto known = completeTileComparison(compare, fullIV, upper, step);
        if (!known) continue;
        b.setInsertionPoint(compare);
        Value value = arith::ConstantIntOp::create(b, compare.getLoc(), *known, 1);
        b.replaceOp(compare, value);
      }
      // A complete mask can eliminate an entire alternative implementation.
      // Charge the surviving fast body, rather than the unsimplified original
      // including that unreachable branch. The masked original remains the
      // remainder; the proposal is bounded before and after simplification.
      SmallVector<Operation *> proposed;
      full.getBody()->walk([&](Operation *op) { proposed.push_back(op); });
      GreedyRewriteConfig config;
      config.setScope(&full.getRegion()).setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
      if (failed(applyOpPatternsGreedily(proposed, patterns, config))) {
        signalPassFailure(); return;
      }
      int64_t added = 0;
      full.getBody()->walk([&](Operation *) { ++added; });
      if (added > budget - duplicatedOperations) { b.eraseOp(full); continue; }
      duplicatedOperations += added;
      b.setInsertionPoint(full);
      // The original signed scf.for is empty for a negative upper bound.
      // Clamp only the split point, so neither new loop invents an iteration.
      Value positive = arith::MaxSIOp::create(b, loop.getLoc(), upper, loop.getLowerBound());
      Value remainder = arith::RemUIOp::create(b, loop.getLoc(), positive, loop.getStep());
      Value split = arith::SubIOp::create(b, loop.getLoc(), positive, remainder);
      full.setUpperBound(split);
      loop.setLowerBound(split);
      loop.getInitArgsMutable().assign(full.getResults());
    }
    for (auto [function, budget] : budgets) {
      function->setAttr("weft.x86.full_tile_operation_budget", b.getI64IntegerAttr(budget));
      function->setAttr("weft.x86.full_tile_added_operations", b.getI64IntegerAttr(duplicated.lookup(function)));
      function->setAttr("weft.x86.full_domain_versions", b.getI64IntegerAttr(versions.lookup(function)));
    }
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86FullTilePass() {
  return std::make_unique<SeparateFullTiles>();
}
