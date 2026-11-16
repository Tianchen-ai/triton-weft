#include "Weft/Target/X86/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
// All FP8 values are exactly representable in f32. Decode the stored bits before
// LLVM conversion: x86 has neither an FP8 register type nor an FP8 conversion ABI.
struct ExpandFP8Widening : OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                 PatternRewriter &b) const override {
    Type source = getElementTypeOrSelf(op.getIn().getType());
    if (!isa<Float8E4M3FNType, Float8E5M2Type>(source)) return failure();
    bool e4m3 = isa<Float8E4M3FNType>(source);
    unsigned mantissa = e4m3 ? 3 : 2;
    Type i8 = b.getI8Type(), i32 = b.getI32Type(), fp32 = b.getF32Type();
    if (auto vector = dyn_cast<VectorType>(op.getIn().getType())) {
      i8 = vector.clone(i8);
      i32 = vector.clone(i32);
      fp32 = vector.clone(fp32);
    }
    Location loc = op.getLoc();
    auto integer = [&](uint32_t value) -> Value {
      TypedAttr attr = b.getI32IntegerAttr(value);
      if (auto vector = dyn_cast<VectorType>(i32)) attr = DenseElementsAttr::get(vector, attr);
      return arith::ConstantOp::create(b, loc, i32, attr);
    };
    Value bits = arith::BitcastOp::create(b, loc, i8, op.getIn());
    bits = arith::ExtUIOp::create(b, loc, i32, bits);
    Value magnitude = arith::AndIOp::create(b, loc, bits, integer(127));
    Value fraction = arith::AndIOp::create(b, loc, magnitude, integer((1u << mantissa) - 1));
    Value exponent = arith::ShRUIOp::create(b, loc, magnitude, integer(mantissa));
    Value biased = arith::AddIOp::create(b, loc, exponent, integer(e4m3 ? 120 : 112));
    Value normal = arith::OrIOp::create(b, loc,
        arith::ShLIOp::create(b, loc, biased, integer(23)),
        arith::ShLIOp::create(b, loc, fraction, integer(23 - mantissa)));
    Value subnormal = arith::UIToFPOp::create(b, loc, fp32, fraction);
    TypedAttr scale = b.getF32FloatAttr(e4m3 ? 0x1p-9 : 0x1p-16);
    if (auto vector = dyn_cast<VectorType>(fp32)) scale = DenseElementsAttr::get(vector, scale);
    subnormal = arith::MulFOp::create(b, loc, subnormal,
        arith::ConstantOp::create(b, loc, fp32, scale));
    Value zeroExponent = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, exponent, integer(0));
    Value decoded = arith::SelectOp::create(b, loc, zeroExponent,
        arith::BitcastOp::create(b, loc, i32, subnormal), normal);
    Value special = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
        e4m3 ? magnitude : exponent, integer(e4m3 ? 127 : 31));
    Value specialBits = integer(0x7fc00000);
    if (!e4m3) {
      Value infinity = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, fraction, integer(0));
      specialBits = arith::SelectOp::create(b, loc, infinity, integer(0x7f800000), specialBits);
    }
    decoded = arith::SelectOp::create(b, loc, special, specialBits, decoded);
    Value sign = arith::ShLIOp::create(b, loc,
        arith::AndIOp::create(b, loc, bits, integer(128)), integer(24));
    decoded = arith::OrIOp::create(b, loc, decoded, sign);
    Value result = arith::BitcastOp::create(b, loc, fp32, decoded);
    if (op.getType() != fp32) {
      if (getElementTypeOrSelf(op.getType()).isF64())
        result = arith::ExtFOp::create(b, loc, op.getType(), result);
      else
        result = arith::TruncFOp::create(b, loc, op.getType(), result);
    }
    b.replaceOp(op, result);
    return success();
  }
};

class LegalizeLowPrecision
    : public PassWrapper<LegalizeLowPrecision, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeLowPrecision)
  StringRef getArgument() const override { return "weft-x86-legalize-low-precision"; }
  StringRef getDescription() const override {
    return "Legalize BF16 arithmetic and FP8 widening without changing stored values or explicit rounding";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    TypeConverter converter;
    arith::populateEmulateUnsupportedFloatsConversions(converter,
        {BFloat16Type::get(context)}, Float32Type::get(context));
    RewritePatternSet arithmetic(context);
    arith::populateEmulateUnsupportedFloatsPatterns(arithmetic, converter);
    ConversionTarget target(*context);
    arith::populateEmulateUnsupportedFloatsLegality(target, converter);
    target.addDynamicallyLegalDialect<math::MathDialect>(
        [&](Operation *op) { return converter.isLegal(op); });
    if (failed(applyPartialConversion(getOperation(), target, std::move(arithmetic)))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet conversions(context);
    arith::populateExpandBFloat16Patterns(conversions);
    conversions.add<ExpandFP8Widening>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(conversions))))
      signalPassFailure();
  }
};

Type payloadType(Type type) {
  if (isa<BFloat16Type, Float8E4M3FNType, Float8E5M2Type>(type))
    return IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth());
  if (auto vector = dyn_cast<VectorType>(type)) return vector.clone(payloadType(vector.getElementType()));
  return type;
}

class LegalizeFloatPayloads
    : public PassWrapper<LegalizeFloatPayloads, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeFloatPayloads)
  StringRef getArgument() const override { return "weft-x86-legalize-float-payloads"; }
  StringRef getDescription() const override {
    return "Carry already legalized low-precision values as exact integer bits through native control flow";
  }
  void runOnOperation() override {
    SmallVector<Operation *> operations;
    bool invalid = false;
    getOperation().walk([&](Operation *op) {
      bool changes = llvm::any_of(op->getOperandTypes(), [](Type t) { return payloadType(t) != t; }) ||
                     llvm::any_of(op->getResultTypes(), [](Type t) { return payloadType(t) != t; });
      if (!changes) return;
      // Numerical operations must already have explicit widening/rounding. A
      // representation pass cannot turn an unlowered arithmetic op into bits.
      if (!isa<arith::ConstantOp, arith::BitcastOp, arith::SelectOp,
               vector::BroadcastOp, vector::ShapeCastOp, vector::ShuffleOp,
               vector::ExtractOp, vector::InsertOp, vector::FromElementsOp,
               vector::ExtractStridedSliceOp, vector::InsertStridedSliceOp,
               vector::BitCastOp, ub::PoisonOp, LLVM::UndefOp, LLVM::BitcastOp,
               LLVM::InsertElementOp, LLVM::ExtractElementOp, LLVM::ShuffleVectorOp,
               cf::BranchOp, cf::CondBranchOp, cf::SwitchOp, func::ReturnOp>(op)) {
        op->emitError("x86 low-precision payload requires prior numerical or memory legalization");
        invalid = true;
      }
      operations.push_back(op);
    });
    if (invalid) return signalPassFailure();
    for (Operation *op : operations) {
      if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        Type type = payloadType(constant.getType());
        if (auto value = dyn_cast<FloatAttr>(constant.getValue()))
          constant.setValueAttr(IntegerAttr::get(type, value.getValue().bitcastToAPInt()));
        else if (auto value = dyn_cast<DenseFPElementsAttr>(constant.getValue()))
          constant.setValueAttr(value.bitcast(cast<VectorType>(type).getElementType()));
        else {
          constant.emitError("x86 low-precision constant requires a literal bit payload");
          return signalPassFailure();
        }
      }
      for (Value result : op->getResults()) result.setType(payloadType(result.getType()));
    }
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      for (Block &block : function.getBody())
        for (BlockArgument arg : block.getArguments()) arg.setType(payloadType(arg.getType()));
      SmallVector<Type> inputs, results;
      for (Type type : function.getFunctionType().getInputs()) inputs.push_back(payloadType(type));
      for (Type type : function.getFunctionType().getResults()) results.push_back(payloadType(type));
      function.setType(FunctionType::get(&getContext(), inputs, results));
    }
    IRRewriter b(&getContext());
    for (Operation *op : operations)
      if (isa<arith::BitcastOp, LLVM::BitcastOp>(op) && op->getOperand(0).getType() == op->getResult(0).getType())
        b.replaceOp(op, op->getOperand(0));
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86LowPrecisionPass() {
  return std::make_unique<LegalizeLowPrecision>();
}

std::unique_ptr<Pass> weft::createX86FloatPayloadPass() {
  return std::make_unique<LegalizeFloatPayloads>();
}
