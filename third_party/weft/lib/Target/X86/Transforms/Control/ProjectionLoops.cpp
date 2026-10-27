#include "Weft/Target/X86/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace {

std::optional<int64_t> unsignedUpperBound(Value value, unsigned depth = 0) {
  if (depth > 16) return {};
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)) && constant.getActiveBits() < 63)
    return constant.getZExtValue();
  if (auto minimum = value.getDefiningOp<arith::MinUIOp>()) {
    auto left = unsignedUpperBound(minimum.getLhs(), depth + 1);
    auto right = unsignedUpperBound(minimum.getRhs(), depth + 1);
    if (left && right) return std::min(*left, *right);
    return left ? left : right;
  }
  return {};
}

// A short loop around local indexing can force an entire SSA vector onto the
// stack for each dynamic extract. Keep its execution guards, but make its
// iterations explicit before folding and vector-projection coalescing. Nested
// guards also expose constant carry updates to the following iteration.
class ExpandProjectionLoops
    : public PassWrapper<ExpandProjectionLoops, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandProjectionLoops)
  StringRef getArgument() const override { return "weft-x86-expand-projection-loops"; }
  StringRef getDescription() const override {
    return "Expose bounded local projection indices while preserving ordered effects and loop carries";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (!loop.getInductionVar().getType().isIndex()) return;
      APInt lower, step;
      if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
          !matchPattern(loop.getStep(), m_ConstantInt(&step)) || step.isZero() || step.isNegative()) return;
      auto upper = unsignedUpperBound(loop.getUpperBound());
      if (!upper || !*upper || step.getActiveBits() >= 63) return;
      int64_t count = (*upper - 1) / step.getZExtValue() + 1;
      if (count > 4) return;
      int64_t operations = 0;
      bool dynamicProjection = false, nestedLoop = false, pure = true;
      loop.getBody()->walk([&](Operation *op) {
        ++operations;
        pure &= isMemoryEffectFree(op);
        if (isa<scf::ForOp, scf::WhileOp>(op)) nestedLoop = true;
        if (auto extract = dyn_cast<LLVM::ExtractElementOp>(op)) {
          APInt position;
          dynamicProjection |= !matchPattern(extract.getPosition(), m_ConstantInt(&position));
        }
      });
      // Expanding effectful access loops also duplicates address/tail programs
      // and can leave their projections dynamic. Limit this transform to local
      // SSA computations; memory traversal keeps its compact ordered loop.
      if (dynamicProjection && pure && !nestedLoop && operations * count <= 12000)
        loops.push_back(loop);
    });
    IRRewriter b(&getContext());
    for (auto loop : loops) {
      APInt step;
      (void)matchPattern(loop.getStep(), m_ConstantInt(&step));
      int64_t increment = step.getZExtValue();
      int64_t count = (*unsignedUpperBound(loop.getUpperBound()) - 1) / increment + 1;
      b.setInsertionPoint(loop);
      auto emit = [&](auto &&emit, int64_t iteration, ValueRange carried) -> SmallVector<Value> {
        Location loc = loop.getLoc();
        Value iv = arith::ConstantIndexOp::create(b, loc, iteration * increment);
        Value condition = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, iv, loop.getUpperBound());
        auto branch = scf::IfOp::create(b, loc, loop.getResultTypes(), condition, true);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
        IRMapping mapping;
        mapping.map(loop.getInductionVar(), iv);
        mapping.map(loop.getRegionIterArgs(), carried);
        for (Operation &op : loop.getBody()->without_terminator()) b.clone(op, mapping);
        SmallVector<Value> next;
        for (Value value : loop.getBody()->getTerminator()->getOperands()) next.push_back(mapping.lookupOrDefault(value));
        if (iteration + 1 < count) next = emit(emit, iteration + 1, next);
        if (branch.getThenRegion().front().mightHaveTerminator())
          b.eraseOp(branch.getThenRegion().front().getTerminator());
        b.setInsertionPointToEnd(&branch.getThenRegion().front());
        scf::YieldOp::create(b, loc, next);
        if (branch.getElseRegion().front().mightHaveTerminator())
          b.eraseOp(branch.getElseRegion().front().getTerminator());
        b.setInsertionPointToEnd(&branch.getElseRegion().front());
        scf::YieldOp::create(b, loc, carried);
        return SmallVector<Value>(branch.getResults());
      };
      auto result = emit(emit, 0, loop.getInitArgs());
      b.replaceOp(loop, result);
    }
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86ProjectionLoopPass() {
  return std::make_unique<ExpandProjectionLoops>();
}
