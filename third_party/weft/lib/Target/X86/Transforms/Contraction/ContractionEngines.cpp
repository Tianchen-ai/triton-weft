#include "Transforms/Contraction/ContractionEngines.h"
#include "Weft/Analysis/LogicalAxes.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir;
namespace wk = weft::kernel;
namespace {
Type element(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return value.getElementType();
  return type;
}
ArrayRef<int64_t> axes(Value value) {
  if (auto type = dyn_cast<wk::ValueType>(value.getType())) return type.getAxisIds().asArrayRef();
  return {};
}
bool matrixAxes(Operation &op) {
  auto over = op.getAttrOfType<DenseI64ArrayAttr>("over").asArrayRef();
  auto domain = [](Value value) -> weft::analysis::LogicalDomain {
    if (auto type = dyn_cast<wk::ValueType>(value.getType()))
      return {type.getShape().asArrayRef(), type.getAxisIds().asArrayRef()};
    return {};
  };
  return weft::analysis::matrixAxes(domain(op.getOperand(0)), domain(op.getOperand(1)),
                                     domain(op.getResult(0)), over).has_value();
}
bool tileExtent(int64_t n, int64_t minimum) { return n >= minimum && (n <= 16 || n % 16 == 0); }
StringRef intent(Operation &op) {
  auto attribute = op.getAttrOfType<StringAttr>("engine");
  return attribute ? attribute.getValue() : "auto";
}
bool allowSubnormalFTZ(Operation &op) {
  auto attr = op.getAttrOfType<BoolAttr>("allow_subnormal_ftz");
  return attr && attr.getValue();
}
bool hasReductionDivisibilityProof(Operation &op, int64_t factor) {
  if (factor <= 1) return true;
  auto over = op.getAttrOfType<DenseI64ArrayAttr>("over").asArrayRef();
  if (over.size() != 1) return false;
  auto la = axes(op.getOperand(0));
  auto pos = llvm::find(la, over[0]);
  if (pos == la.end()) return false;
  size_t kIdx = static_cast<size_t>(pos - la.begin());
  bool proven = false;
  if (auto block = op.getBlock()) {
    for (Operation &prev : *block) {
      if (&prev == &op) break;
      if (auto fact = dyn_cast<wk::ValueFactOp>(prev)) {
        if (fact.getKind() == "divisibility" && !fact.getInputs().empty() &&
            fact.getInputs().back() == op.getOperand(0) &&
            kIdx < fact.getValues().size() &&
            fact.getValues()[kIdx] % factor == 0)
          proven = true;
      }
      if (auto assume = dyn_cast<wk::AssumeOp>(prev)) {
        if (auto cmp = assume.getCondition().getDefiningOp<arith::CmpIOp>();
            cmp && cmp.getPredicate() == arith::CmpIPredicate::eq) {
          APInt zero;
          if (matchPattern(cmp.getRhs(), m_ConstantInt(&zero)) && zero.isZero()) {
            if (auto rem = cmp.getLhs().getDefiningOp<arith::RemUIOp>()) {
              APInt div;
              if (matchPattern(rem.getRhs(), m_ConstantInt(&div)) &&
                  div.getSExtValue() > 0 && div.getSExtValue() % factor == 0)
                proven = true;
            }
          }
        }
      }
    }
  }
  return proven;
}
} // namespace

bool weft::x86::matrixReductionFirst(Operation &op, unsigned operand) {
  auto over = op.getAttrOfType<DenseI64ArrayAttr>("over").asArrayRef();
  auto ids = axes(op.getOperand(operand));
  return over.size() == 1 && ids.size() == 2 && ids[0] == over[0];
}

SmallVector<int64_t> weft::x86::matrixOperandShape(Operation &op, unsigned operand,
                                                   ArrayRef<int64_t> shape) {
  if (matrixReductionFirst(op, operand)) return {shape[1], shape[0]};
  return SmallVector<int64_t>(shape);
}

weft::x86::BrgemmGeometry weft::x86::describeBrgemmGeometry(
    Operation &op, ArrayRef<int64_t> left, ArrayRef<int64_t> right, ArrayRef<int64_t> output) {
  BrgemmGeometry g;
  auto overAttr = op.getAttrOfType<DenseI64ArrayAttr>("over");
  if (!overAttr) return g;
  auto over = overAttr.asArrayRef();
  auto la = axes(op.getOperand(0));
  auto ra = axes(op.getOperand(1));
  auto oa = axes(op.getResult(0));
  if (over.size() == 1 && la.size() == 2 && ra.size() == 2 && oa.size() == 2 &&
      left.size() == 2 && right.size() == 2 && output.size() == 2) {
    bool lrf = (la[0] == over[0]);
    if (!lrf && la[1] != over[0]) return g;
    bool rrf = (ra[0] == over[0]);
    if (!rrf && ra[1] != over[0]) return g;
    int64_t mAxis = la[lrf ? 1 : 0];
    int64_t nAxis = ra[rrf ? 1 : 0];
    if (mAxis == nAxis || oa[0] != mAxis || oa[1] != nAxis) return g;
    g.valid = true;
    g.leftReductionFirst = lrf;
    g.rightReductionFirst = rrf;
    g.staticM = left[lrf ? 1 : 0];
    g.staticK = left[lrf ? 0 : 1];
    g.staticN = right[rrf ? 1 : 0];
    g.kBlock = 1;
    return g;
  }
  if (over.size() == 1 && la.size() == 2 && ra.size() == 1 && oa.size() == 1 &&
      left.size() == 2 && right.size() == 1 && output.size() == 1) {
    bool lrf = (la[0] == over[0]);
    if ((!lrf && la[1] != over[0]) || ra[0] != over[0]) return g;
    int64_t mAxis = la[lrf ? 1 : 0];
    if (oa[0] != mAxis) return g;
    g.valid = true;
    g.vectorRight = true;
    g.leftReductionFirst = lrf;
    g.rightReductionFirst = true;
    g.staticM = left[lrf ? 1 : 0];
    g.staticK = left[lrf ? 0 : 1];
    g.staticN = 1;
    g.kBlock = 1;
    return g;
  }
  if (over.size() == 2 && la.size() == 3 && ra.size() == 3 && oa.size() == 2 &&
      left.size() == 3 && right.size() == 3 && output.size() == 2) {
    if (la[1] != over[0] || la[2] != over[1] ||
        ra[0] != over[0] || ra[2] != over[1])
      return g;
    int64_t mAxis = la[0], nAxis = ra[1];
    if (mAxis == nAxis || oa[0] != mAxis || oa[1] != nAxis) return g;
    if (left[2] <= 0 || left[2] != right[2]) return g;
    g.valid = true;
    g.packed3D = true;
    g.leftReductionFirst = false;
    g.rightReductionFirst = true;
    g.kBlock = left[2];
    g.staticM = left[0];
    g.staticN = right[1];
    g.staticK = left[1] > 0 ? left[1] * left[2] : -1;
    return g;
  }
  return g;
}

bool weft::x86::integerMatrixGeometry(Operation &op, const X86Target &target,
                                      ArrayRef<int64_t> left, ArrayRef<int64_t> right) {
  if (!target.hasFeature("amx-int8") || !element(op.getResult(0).getType()).isInteger(32) || !matrixAxes(op))
    return false;
  unsigned wideInputs = 0;
  for (Value input : op.getOperands()) {
    auto type = dyn_cast<IntegerType>(element(input.getType()));
    if (!type || type.getWidth() > 16) return false;
    wideInputs += type.getWidth() > 8;
  }
  auto lhs = matrixOperandShape(op, 0, left), rhs = matrixOperandShape(op, 1, right);
  return wideInputs <= 1 && tileExtent(lhs[0], 1) && tileExtent(rhs[0], 1) &&
         lhs[1] >= 16 && lhs[1] % 16 == 0 && lhs[1] == rhs[1];
}

FailureOr<weft::x86::ContractionEngine> weft::x86::selectContractionEngine(
    Operation &op, const X86Target &target, StringRef contractionBackend,
    ContractionStorage storage, bool arrayResult,
    ArrayRef<int64_t> left, ArrayRef<int64_t> right, ArrayRef<int64_t> output,
    bool emitDiagnostics) {
  StringRef req = intent(op);
  auto fail = [&](const Twine &msg) -> FailureOr<ContractionEngine> {
    if (emitDiagnostics) op.emitError(msg);
    return failure();
  };

  if (contractionBackend == "native") {
    if (req.starts_with("onednn-brgemm"))
      return fail("contraction_backend=native conflicts with explicit engine=" + req);
    if (req != "matrix")
      return ContractionEngine::Vector;
  }

  bool wantVendor = req.starts_with("onednn-brgemm") ||
                    (contractionBackend == "onednn-brgemm" && req != "vector" && req != "matrix");
  if (wantVendor) {
    const auto &brgemm = probeOneDNNBrgemm();
    auto geom = describeBrgemmGeometry(op, left, right, output);
    Type lhsElem = element(op.getOperand(0).getType());
    Type rhsElem = element(op.getOperand(1).getType());
    Type resElem = element(op.getResult(0).getType());
    bool kDiv2 = (geom.packed3D && geom.kBlock % 2 == 0) ||
                 (geom.staticK > 0 && geom.staticK % 2 == 0) ||
                 hasReductionDivisibilityProof(op, 2);
    bool kDiv4 = (geom.packed3D && geom.kBlock % 4 == 0) ||
                 (geom.staticK > 0 && geom.staticK % 4 == 0) ||
                 hasReductionDivisibilityProof(op, 4);

    if (brgemm.available && target.vectorBits == 512 && geom.valid) {
      if (lhsElem.isF32() && rhsElem.isF32() && resElem.isF32() && !geom.packed3D) {
        if (brgemm.f32 && (req == "auto" || req == "onednn-brgemm" || req == "onednn-brgemm-f32"))
          return ContractionEngine::OneDNNBrgemmF32;
      } else if (lhsElem.isF16() && rhsElem.isF16() && resElem.isF32() && !geom.packed3D) {
        if (brgemm.f16 && target.hasHostFeature("avx512fp16") &&
            (req == "auto" || req == "onednn-brgemm" || req == "onednn-brgemm-f16"))
          return ContractionEngine::OneDNNBrgemmF16;
        if (brgemm.f32 && (req == "auto" || req == "onednn-brgemm" || req == "onednn-brgemm-f32"))
          return ContractionEngine::OneDNNBrgemmF32;
      } else if (lhsElem.isBF16() && rhsElem.isBF16() && resElem.isF32()) {
        if (req == "onednn-brgemm-bf16") {
          if (!allowSubnormalFTZ(op))
            return fail("engine=onednn-brgemm-bf16 requires allow_subnormal_ftz = true");
          if (!kDiv2 || (geom.packed3D && geom.kBlock != 2))
            return fail("engine=onednn-brgemm-bf16 requires even reduction extent K");
          if (!brgemm.bf16)
            return fail("engine=onednn-brgemm-bf16 is unavailable on this x86 host");
          return ContractionEngine::OneDNNBrgemmBF16;
        }
        if (allowSubnormalFTZ(op) && brgemm.bf16 && kDiv2 &&
            (!geom.packed3D || geom.kBlock == 2) &&
            (req == "auto" || req == "onednn-brgemm"))
          return ContractionEngine::OneDNNBrgemmBF16;
        if (brgemm.f32 && (req == "auto" || req == "onednn-brgemm" || req == "onednn-brgemm-f32"))
          return ContractionEngine::OneDNNBrgemmF32;
      } else if (auto li = dyn_cast<IntegerType>(lhsElem)) {
        auto ri = dyn_cast<IntegerType>(rhsElem);
        auto oi = dyn_cast<IntegerType>(resElem);
        if (ri && oi && li.getWidth() == 8 && ri.getWidth() == 8 &&
            oi.getWidth() == 32 && oi.isSigned() && ri.isSigned()) {
          if (li.isSigned()) {
            if (req == "onednn-brgemm-s8" && !kDiv4)
              return fail("engine=onednn-brgemm-s8 requires reduction extent K divisible by 4");
            if (brgemm.s8 && kDiv4 && (!geom.packed3D || geom.kBlock == 4) &&
                (req == "auto" || req == "onednn-brgemm" || req == "onednn-brgemm-s8"))
              return ContractionEngine::OneDNNBrgemmS8;
          } else if (li.isUnsigned()) {
            if (req == "onednn-brgemm-u8s8" && !kDiv4)
              return fail("engine=onednn-brgemm-u8s8 requires reduction extent K divisible by 4");
            if (brgemm.u8s8 && kDiv4 && (!geom.packed3D || geom.kBlock == 4) &&
                (req == "auto" || req == "onednn-brgemm" || req == "onednn-brgemm-u8s8"))
              return ContractionEngine::OneDNNBrgemmU8S8;
          }
        }
      }
    }
    if (req.starts_with("onednn-brgemm")) {
      std::string detail = !brgemm.available ? (" (" + brgemm.reason + ")") : "";
      return fail("requested " + req +
                  " has no legal oneDNN BRGeMM ukernel for these types, axes, extents, or target capabilities" +
                  detail);
    }
  }

  if (req == "vector") return ContractionEngine::Vector;
  bool integer = integerMatrixGeometry(op, target, left, right);
  if (integer && arrayResult) {
    int64_t outputs = std::accumulate(output.begin(), output.end(), int64_t(1),
                                      std::multiplies<int64_t>());
    integer = (storage.leftArray || storage.leftVector) && element(op.getOperand(0).getType()).isInteger(8) &&
        ((storage.rightArray && cast<IntegerType>(element(op.getOperand(1).getType())).getWidth() >= 8) ||
         (storage.rightVector && outputs <= 4096));
  }
  if (integer) return ContractionEngine::MatrixInteger;
  bool bfloat = arrayResult && target.hasFeature("amx-bf16") && matrixAxes(op) &&
      element(op.getResult(0).getType()).isF32() &&
      element(op.getOperand(0).getType()).isBF16() && element(op.getOperand(1).getType()).isBF16() &&
      storage.leftArray && storage.rightArray;
  if (bfloat) {
    auto lhs = matrixOperandShape(op, 0, left), rhs = matrixOperandShape(op, 1, right);
    bfloat = tileExtent(lhs[0], 1) && tileExtent(rhs[0], 1) &&
             lhs[1] >= 32 && lhs[1] % 32 == 0 && lhs[1] == rhs[1];
  }
  if (req == "matrix" && bfloat)
    return fail("engine=matrix cannot use the guarded BF16 route: its full numerical contract may require vector execution for runtime input values");
  if (bfloat) return ContractionEngine::MatrixBFloat;
  if (req == "matrix")
    return fail("engine=matrix has no legal x86 matrix implementation for these types, axes, extents, storage and target features");
  return ContractionEngine::Vector;
}

void weft::x86::recordContractionEngine(ModuleOp module, Operation &source, ContractionEngine engine) {
  Builder b(module.getContext());
  SmallVector<Attribute> records;
  if (auto existing = module->getAttrOfType<ArrayAttr>("weft.x86.contractions"))
    records.append(existing.begin(), existing.end());
  std::string location;
  llvm::raw_string_ostream stream(location);
  source.getLoc().print(stream);
  StringRef selectedName = "vector";
  StringRef family = "simd";
  if (isOneDNNBrgemm(engine)) {
    selectedName = "vendor";
    switch (engine) {
    case ContractionEngine::OneDNNBrgemmF32: family = "onednn-brgemm-f32"; break;
    case ContractionEngine::OneDNNBrgemmF16: family = "onednn-brgemm-f16"; break;
    case ContractionEngine::OneDNNBrgemmBF16: family = "onednn-brgemm-bf16"; break;
    case ContractionEngine::OneDNNBrgemmS8: family = "onednn-brgemm-s8"; break;
    case ContractionEngine::OneDNNBrgemmU8S8: family = "onednn-brgemm-u8s8"; break;
    default: break;
    }
  } else if (engine != ContractionEngine::Vector) {
    selectedName = "matrix";
    family = engine == ContractionEngine::MatrixInteger ? "amx-int8" : "amx-bf16";
  }
  records.push_back(b.getDictionaryAttr({
      b.getNamedAttr("source", b.getStringAttr(location)),
      b.getNamedAttr("requested_engine", b.getStringAttr(intent(source))),
      b.getNamedAttr("selected_engine", b.getStringAttr(selectedName)),
      b.getNamedAttr("instruction_family", b.getStringAttr(family)),
      b.getNamedAttr("conditional", b.getBoolAttr(engine == ContractionEngine::MatrixBFloat)),
      b.getNamedAttr("selection_rule", b.getStringAttr("author constraint; vendor/matrix legality; explicit numerical contract"))}));
  module->setAttr("weft.x86.contractions", b.getArrayAttr(records));
}
