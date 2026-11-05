#include "Weft/Dialect/Kernel/Transforms/SourceSpecialization.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace {
class PushCoordinateRename : public OpRewritePattern<weft::kernel::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(weft::kernel::ReshapeOp reshape,
                               PatternRewriter &rewriter) const override {
    namespace k = weft::kernel;
    auto input = reshape.getInput().getType();
    auto output = reshape.getResult().getType();
    if (input == output && llvm::equal(reshape.getOrder(), input.getAxisIds().asArrayRef())) {
      rewriter.replaceOp(reshape, reshape.getInput());
      return success();
    }
    if (input.getShape() != output.getShape() ||
        !llvm::equal(reshape.getOrder(), input.getAxisIds().asArrayRef()))
      return failure();
    Operation *producer = reshape.getInput().getDefiningOp();
    if (!producer) return failure();
    bool elementwise = isa<k::IotaOp, k::UnaryOp, k::BinaryOp, k::CompareOp,
                           k::SelectOp, k::CastOp, k::BitcastOp, k::WidenOp, k::NarrowOp>(producer);
    auto extract = dyn_cast<k::ExtractOp>(producer);
    auto lookup = dyn_cast<k::LookupOp>(producer);
    if (!elementwise && !extract && !lookup) return failure();
    if (!reshape.getInput().hasOneUse() && !extract && !isa<k::IotaOp>(producer))
      return failure();
    // An all/domain selector also carries the owner's coordinate identity.
    // Keep those boundaries explicit; only gather result coordinates commute.
    if (extract && llvm::any_of(extract.getSelectors(), [](Attribute attr) {
          return cast<StringAttr>(attr).getValue() != "gather" &&
                 cast<StringAttr>(attr).getValue() != "index";
        })) return failure();
    llvm::DenseMap<int64_t, int64_t> mapping;
    for (auto [oldAxis, newAxis] : llvm::zip(input.getAxisIds().asArrayRef(), output.getAxisIds().asArrayRef()))
      mapping[oldAxis] = newAxis;
    for (auto [position, operand] : llvm::enumerate(producer->getOperands())) {
      if ((extract || lookup) && position == 0) continue;
      if (auto type = dyn_cast<k::ValueType>(operand.getType()))
        for (int64_t axis : type.getAxisIds().asArrayRef())
          if (!mapping.contains(axis)) return failure();
    }
    SmallVector<Value> operands;
    for (auto [position, operand] : llvm::enumerate(producer->getOperands())) {
      auto type = dyn_cast<k::ValueType>(operand.getType());
      if (!type || ((extract || lookup) && position == 0)) {
        operands.push_back(operand);
        continue;
      }
      SmallVector<int64_t> axes;
      for (int64_t axis : type.getAxisIds().asArrayRef()) {
        auto it = mapping.find(axis);
        axes.push_back(it->second);
      }
      auto renamed = k::ValueType::get(getContext(), type.getElementType(), type.getShape(),
                                      rewriter.getDenseI64ArrayAttr(axes));
      if (renamed == type) operands.push_back(operand);
      else {
        Value existing;
        for (Operation *user : operand.getUsers()) {
          auto alias = dyn_cast<k::ReshapeOp>(user);
          if (alias && alias.getResult().getType() == renamed &&
              alias.getOrderAttr() == type.getAxisIds() &&
              alias->getBlock() == reshape->getBlock() && alias->isBeforeInBlock(reshape)) {
            existing = alias.getResult();
            break;
          }
        }
        operands.push_back(existing ? existing : k::ReshapeOp::create(rewriter, reshape.getLoc(),
            renamed, operand, type.getAxisIds()).getResult());
      }
    }
    OperationState state(producer->getLoc(), producer->getName());
    state.addOperands(operands);
    state.addTypes(output);
    state.addAttributes(producer->getAttrs());
    rewriter.replaceOp(reshape, rewriter.create(state)->getResults());
    if (producer->use_empty()) rewriter.eraseOp(producer);
    return success();
  }
};

class NormalizeCanonicalControl
    : public PassWrapper<NormalizeCanonicalControl, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeCanonicalControl)
  StringRef getArgument() const override { return "weft-normalize-canonical-control"; }
  StringRef getDescription() const override {
    return "Inline ordinary single-exit helpers and normalize SCF carries before physical lowering";
  }
  void runOnOperation() override {
    SmallVector<weft::kernel::HelperOp> helpers;
    getOperation().walk<WalkOrder::PostOrder>([&](weft::kernel::HelperOp helper) {
      helpers.push_back(helper);
    });
    for (auto helper : helpers) {
      if (helper.getNoinline() || !llvm::hasSingleElement(helper.getBody())) continue;
      Block &body = helper.getBody().front();
      if (body.empty()) continue;
      auto exit = dyn_cast<weft::kernel::HelperReturnOp>(body.back());
      if (!exit) continue;
      bool nestedExit = false;
      helper.getBody().walk([&](weft::kernel::HelperReturnOp ret) {
        if (ret != exit && ret->getParentOfType<weft::kernel::HelperOp>() == helper)
          nestedExit = true;
      });
      if (nestedExit) continue;
      helper.replaceAllUsesWith(exit.getValues());
      exit.erase();
      helper->getBlock()->getOperations().splice(helper->getIterator(), body.getOperations());
      helper.erase();
    }
    SmallVector<scf::ForOp> unrolled;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.unroll")) unrolled.push_back(loop);
    });
    for (auto loop : unrolled) {
      loop->removeAttr("weft.unroll");
      if (failed(loopUnrollFull(loop))) {
        loop.emitError("source unroll requires a statically bounded SCF loop");
        signalPassFailure();
        return;
      }
    }
    SmallVector<Operation *> unrollBounds;
    getOperation().walk([&](Operation *operation) {
      if (isa<arith::AddIOp, arith::MulIOp>(operation)) unrollBounds.push_back(operation);
    });
    GreedyRewriteConfig foldBounds;
    foldBounds.setStrictness(GreedyRewriteStrictness::ExistingOps);
    if (failed(applyOpPatternsGreedily(unrollBounds, RewritePatternSet(&getContext()), foldBounds))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
    scf::WhileOp::getCanonicalizationPatterns(patterns, &getContext());
    patterns.add<PushCoordinateRename>(&getContext());
    // Dead control carries can be the only extra consumer of a coordinate
    // rename. Revisit both in the same bounded worklist, so deleting a carry
    // exposes the producer to the existing single-consumer rewrite. Folding
    // remains disabled here: Canonical domains retain their explicit count
    // relation, including ceildiv(extent, 1), until physical lowering.
    GreedyRewriteConfig configuration;
    configuration.enableFolding(false).enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), configuration))) {
      signalPassFailure();
      return;
    }
    IRRewriter rewriter(&getContext());
    // CSE may share pure numerical expressions, but two source declarations
    // still represent two births. Keep their identities distinct during the
    // generic transform, without adding an attribute to the resulting IR.
    int64_t birth = 0;
    getOperation().walk([&](Operation *op) {
      if (isa<weft::kernel::NewOp, weft::kernel::MaterializeOp>(op))
        op->setAttr("weft.cse_birth", rewriter.getI64IntegerAttr(birth++));
    });
    DominanceInfo dominance(getOperation());
    eliminateCommonSubExpressions(rewriter, dominance, getOperation());
    getOperation().walk([](Operation *op) { op->removeAttr("weft.cse_birth"); });
  }
};
}

std::unique_ptr<Pass> weft::createNormalizeCanonicalControlPass() {
  return std::make_unique<NormalizeCanonicalControl>();
}
