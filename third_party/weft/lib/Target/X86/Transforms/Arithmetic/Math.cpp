#include "Weft/Target/X86/X86Compiler.h"
#include "Transforms/Arithmetic/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

using namespace mlir;
namespace {
Operation *singleReadOrigin(Value value) {
  SmallVector<Value> pending{value};
  llvm::SmallPtrSet<Operation *, 32> visited;
  Operation *origin = nullptr;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    Operation *op = current.getDefiningOp();
    if (!op || !visited.insert(op).second) continue;
    if (visited.size() > 64) return nullptr;
    if (isa<LLVM::LoadOp, LLVM::MaskedLoadOp, LLVM::masked_gather, memref::LoadOp>(op)) {
      if (origin && origin != op) return nullptr;
      origin = op;
      continue;
    }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      APInt condition;
      if (!matchPattern(branch.getCondition(), m_ConstantInt(&condition))) {
        // A conditional read still defines one saved SSA snapshot. Two
        // projections of that same result must stay packed together.
        if (origin && origin != op) return nullptr;
        origin = op;
        continue;
      }
      Region &region = condition.isZero() ? branch.getElseRegion() : branch.getThenRegion();
      if (region.empty()) return nullptr;
      pending.push_back(region.front().getTerminator()->getOperand(cast<OpResult>(current).getResultNumber()));
      continue;
    }
    if (!isMemoryEffectFree(op) || op->getNumRegions()) return nullptr;
    llvm::append_range(pending, op->getOperands());
  }
  return origin;
}

std::optional<unsigned> scalarProjectionLane(Operation *operation) {
  if (auto extract = dyn_cast<LLVM::ExtractElementOp>(operation)) {
    APInt position;
    if (matchPattern(extract.getPosition(), m_ConstantInt(&position)) &&
        position.getLimitedValue(2) < 2)
      return position.getZExtValue();
  }
  if (auto extract = dyn_cast<vector::ExtractOp>(operation)) {
    if (!extract.hasDynamicPosition() && extract.getStaticPosition().size() == 1 &&
        !isa<VectorType>(extract.getType())) {
      int64_t lane = extract.getStaticPosition().front();
      if (lane >= 0 && lane < 2) return static_cast<unsigned>(lane);
    }
  }
  return std::nullopt;
}

class LegalizeHalfCarriers : public PassWrapper<LegalizeHalfCarriers, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeHalfCarriers)
  StringRef getArgument() const override { return "weft-x86-legalize-half-carriers"; }
  StringRef getDescription() const override { return "Avoid partial-register conversion dependencies between independent half snapshots"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, vector::VectorDialect, ub::UBDialect>();
  }
  void runOnOperation() override {
    SmallVector<arith::ExtFOp> conversions;
    getOperation().walk([&](arith::ExtFOp op) { conversions.push_back(op); });
    IRRewriter b(&getContext());
    for (auto op : conversions) {
      auto function = op->getParentOfType<func::FuncOp>();
      auto cpu = function->getAttrOfType<StringAttr>("weft.x86.cpu");
      weft::X86Target target;
      std::string error;
      if (!cpu || !weft::parseX86Target(cpu.str(), "sysv", 0, target, error)) continue;
      auto input = dyn_cast<VectorType>(op.getIn().getType());
      if (!input || input.isScalable() || input.getRank() != 1 ||
          !input.getElementType().isF16() || input.getNumElements() != 2) continue;
      auto output = cast<VectorType>(op.getType());
      if (!output.getElementType().isF32() && !output.getElementType().isF64()) continue;
      auto select = op.getIn().getDefiningOp<arith::SelectOp>();
      if (!select) continue;
      auto left = select.getTrueValue().getDefiningOp<vector::BroadcastOp>();
      auto right = select.getFalseValue().getDefiningOp<vector::BroadcastOp>();
      auto constant = select.getCondition().getDefiningOp<arith::ConstantOp>();
      auto mask = constant ? dyn_cast<DenseIntElementsAttr>(constant.getValue()) : DenseIntElementsAttr();
      if (!left || !right || !left.getSource().getType().isF16() || !right.getSource().getType().isF16() ||
          !mask || mask.isSplat()) continue;
      Operation *a = singleReadOrigin(left.getSource()), *c = singleReadOrigin(right.getSource());
      if (!a || !c || a == c) continue;
      if (target.vectorBits != 512) continue;
      b.setInsertionPoint(op);
      // Both constants occur in the mask, so both scalar conversions were
      // already demanded. Reads and explicit narrowing stay at their origins.
      SmallVector<std::pair<Operation *, unsigned>> projections;
      bool scalarOnly = !op->use_empty();
      for (Operation *user : op->getUsers()) {
        auto lane = scalarProjectionLane(user);
        if (!lane) { scalarOnly = false; break; }
        projections.emplace_back(user, *lane);
      }
      if (target.hasFeature("avx512fp16") && scalarOnly) {
        // No consumer needs a packed result. Two native scalar conversions
        // keep independent load/conversion chains instead of serializing an
        // insert, packed conversion and extraction on one XMM carrier.
        SmallVector<Value, 2> sources;
        for (APInt enabled : mask.getValues<APInt>())
          sources.push_back(enabled.isZero() ? right.getSource() : left.getSource());
        Value widened[2];
        for (auto [user, lane] : projections) {
          if (!widened[lane])
            widened[lane] = output.getElementType().isF32()
                ? weft::widenX86ScalarHalf(getOperation(), b, op.getLoc(), sources[lane], target)
                : arith::ExtFOp::create(b, op.getLoc(), output.getElementType(), sources[lane]).getResult();
          b.replaceOp(user, widened[lane]);
        }
        b.eraseOp(op);
        continue;
      }
      if (target.hasFeature("f16c")) {
        // Define the complete machine input through integer bits. The two
        // independent scalar snapshots must not share undefined XMM lanes,
        // but widening each one separately also loses the packed conversion.
        auto carrierType = VectorType::get({8}, b.getI16Type());
        Value carrier = arith::ConstantOp::create(b, op.getLoc(), carrierType,
            DenseIntElementsAttr::get(carrierType, APInt(16, 0)));
        for (auto [lane, enabled] : llvm::enumerate(mask.getValues<APInt>())) {
          Value source = enabled.isZero() ? right.getSource() : left.getSource();
          Value bits = arith::BitcastOp::create(b, op.getLoc(), b.getI16Type(), source);
          carrier = vector::InsertOp::create(b, op.getLoc(), bits, carrier,
              ArrayRef<int64_t>{static_cast<int64_t>(lane)});
        }
        auto fp = VectorType::get({4}, b.getF32Type());
        StringRef name = "llvm.x86.vcvtph2ps.128";
        auto function = getOperation().lookupSymbol<func::FuncOp>(name);
        if (!function) {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToEnd(getOperation().getBody());
          function = func::FuncOp::create(b, op.getLoc(), name,
              b.getFunctionType({carrierType}, {fp}));
          function.setPrivate();
        }
        Value converted = func::CallOp::create(b, op.getLoc(), function, ValueRange{carrier}).getResult(0);
        Value result = vector::ExtractStridedSliceOp::create(b, op.getLoc(), converted,
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{2}, ArrayRef<int64_t>{1});
        if (output.getElementType().isF64()) result = arith::ExtFOp::create(b, op.getLoc(), output, result);
        b.replaceOp(op, result);
        continue;
      }
      auto widen = [&](Value value) -> Value {
        Value result = weft::widenX86ScalarHalf(getOperation(), b, op.getLoc(), value, target);
        if (output.getElementType().isF64()) result = arith::ExtFOp::create(b, op.getLoc(), b.getF64Type(), result);
        return vector::BroadcastOp::create(b, op.getLoc(), output, result);
      };
      b.replaceOpWithNewOp<arith::SelectOp>(op, select.getCondition(), widen(left.getSource()), widen(right.getSource()));
    }
  }
};

class ExpandX86Math : public PassWrapper<ExpandX86Math, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandX86Math)
  StringRef getArgument() const override { return "weft-x86-expand-transcendentals"; }
  StringRef getDescription() const override {
    return "Materialize vector exponential arithmetic with explicit range and special-value handling";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    SmallVector<math::ExpOp> operations;
    getOperation().walk([&](math::ExpOp op) {
      Type element = op.getType();
      if (auto vector = dyn_cast<VectorType>(element)) element = vector.getElementType();
      if (element.isF32()) operations.push_back(op);
    });
    IRRewriter b(&getContext());
    for (auto op : operations) {
      b.setInsertionPoint(op);
      Location loc = op.getLoc();
      Type type = op.getType();
      auto constant = [&](double value) -> Value {
        auto scalar = b.getF32FloatAttr(value);
        TypedAttr attr = scalar;
        if (auto vector = dyn_cast<VectorType>(type)) attr = DenseElementsAttr::get(vector, scalar);
        return arith::ConstantOp::create(b, loc, type, attr);
      };
      Value x = op.getOperand();
      Value low = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, x, constant(-80.0));
      // The SDK approximation flushes tiny results. Rescale low inputs by an
      // exact binary factor so its polynomial operates on normal values and
      // the final multiply produces the actual subnormal result.
      Value shifted = arith::AddFOp::create(b, loc, x, constant(44.361419677734375));
      shifted = arith::AddFOp::create(b, loc, shifted, constant(-1.21897875e-7));
      Value bounded = arith::SelectOp::create(b, loc, low, shifted, x);
      Value approximated = math::ExpOp::create(b, loc, bounded);
      Value scaled = arith::MulFOp::create(b, loc, approximated, constant(0x1p-64));
      Value output = arith::SelectOp::create(b, loc, low, scaled, approximated);
      // The rounded f32 input at this boundary already overflows expf. Retain
      // the overflow and NaN contract independently of the polynomial's clamp.
      Value overflow = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGE, x, constant(88.72283935546875));
      output = arith::SelectOp::create(b, loc, overflow, constant(std::numeric_limits<float>::infinity()), output);
      Value nan = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UNO, x, x);
      output = arith::SelectOp::create(b, loc, nan, x, output);
      b.replaceOp(op, output);
    }
    RewritePatternSet patterns(&getContext());
    populateMathPolynomialApproximationPatterns(patterns,
        [](StringRef name) { return name == math::ExpOp::getOperationName(); });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86MathPass() { return std::make_unique<ExpandX86Math>(); }
std::unique_ptr<Pass> weft::createX86HalfCarrierPass() { return std::make_unique<LegalizeHalfCarriers>(); }

Value weft::widenX86ScalarHalf(ModuleOp module, OpBuilder &b, Location loc,
                             Value input, const X86Target &target) {
  if (target.hasFeature("avx512fp16")) {
    auto fp = VectorType::get({4}, b.getF32Type());
    auto hp = VectorType::get({8}, b.getF16Type());
    Value zero = arith::ConstantOp::create(b, loc, fp, DenseElementsAttr::get(fp, b.getF32FloatAttr(0)));
    // Only the low half is consumed by the scalar instruction. Its remaining
    // lanes need no initialization; the independent, defined merge operand
    // below owns the result's upper lanes.
    Value source = ub::PoisonOp::create(b, loc, hp);
    source = vector::InsertOp::create(b, loc, input, source, ArrayRef<int64_t>{0});
    Value mask = arith::ConstantOp::create(b, loc, b.getI8Type(), b.getI8IntegerAttr(1));
    Value rounding = arith::ConstantOp::create(b, loc, b.getI32Type(), b.getI32IntegerAttr(4));
    StringRef name = "llvm.x86.avx512fp16.mask.vcvtsh2ss.round";
    auto function = module.lookupSymbol<func::FuncOp>(name);
    if (!function) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToEnd(module.getBody());
      function = func::FuncOp::create(b, loc, name,
          b.getFunctionType({fp, hp, fp, b.getI8Type(), b.getI32Type()}, {fp}));
      function.setPrivate();
    }
    function->setAttr("weft.x86.preserves_tile_state", b.getUnitAttr());
    // Upper lanes are not part of the scalar value. Define their merge source
    // independently so the saved half has one numeric use and a legal original
    // load can fold into the scalar conversion's m16 operand.
    Value result = func::CallOp::create(b, loc, function,
        ValueRange{zero, source, zero, mask, rounding}).getResult(0);
    return vector::ExtractOp::create(b, loc, result, ArrayRef<int64_t>{0});
  }
  Value bits = arith::BitcastOp::create(b, loc, b.getI16Type(), input);
  Value carrier = vector::BroadcastOp::create(b, loc, VectorType::get({16}, b.getI16Type()), bits);
  auto fp = VectorType::get({16}, b.getF32Type());
  Value zero = arith::ConstantOp::create(b, loc, fp, DenseElementsAttr::get(fp, b.getF32FloatAttr(0)));
  Value enabled = arith::ConstantIntOp::create(b, loc, -1, 16);
  Value mode = arith::ConstantIntOp::create(b, loc, 4, 32);
  StringRef name = "llvm.x86.avx512.mask.vcvtph2ps.512";
  auto function = module.lookupSymbol<func::FuncOp>(name);
  if (!function) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    function = func::FuncOp::create(b, loc, name,
        b.getFunctionType({carrier.getType(), fp, b.getI16Type(), b.getI32Type()}, {fp}));
    function.setPrivate();
  }
  function->setAttr("weft.x86.preserves_tile_state", b.getUnitAttr());
  Value widened = func::CallOp::create(b, loc, function, ValueRange{carrier, zero, enabled, mode}).getResult(0);
  return vector::ExtractOp::create(b, loc, widened, ArrayRef<int64_t>{0});
}
