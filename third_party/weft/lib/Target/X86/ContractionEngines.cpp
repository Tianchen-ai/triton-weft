#include "ContractionEngines.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/IR/Builders.h"
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
  auto left = axes(op.getOperand(0)), right = axes(op.getOperand(1)), output = axes(op.getResult(0));
  return over.size() == 1 && left.size() == 2 && right.size() == 2 && output.size() == 2 &&
         ((left[0] == output[0] && left[1] == over[0]) ||
          (left[1] == output[0] && left[0] == over[0])) &&
         ((right[0] == output[1] && right[1] == over[0]) ||
          (right[1] == output[1] && right[0] == over[0]));
}
bool tileExtent(int64_t n, int64_t minimum) { return n >= minimum && (n <= 16 || n % 16 == 0); }
StringRef intent(Operation &op) {
  auto attribute = op.getAttrOfType<StringAttr>("engine");
  return attribute ? attribute.getValue() : "auto";
}
}

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
  return wideInputs <= 1 && tileExtent(lhs[0], 8) && tileExtent(rhs[0], 8) &&
         lhs[1] >= (wideInputs ? 128 : 16) && lhs[1] % 16 == 0 && lhs[1] == rhs[1];
}

FailureOr<weft::x86::ContractionEngine> weft::x86::selectContractionEngine(
    Operation &op, const X86Target &target, ContractionStorage storage, bool arrayResult,
    ArrayRef<int64_t> left, ArrayRef<int64_t> right, ArrayRef<int64_t> output) {
  if (intent(op) == "vector") return ContractionEngine::Vector;
  if (arrayResult && (matrixReductionFirst(op, 0) || matrixReductionFirst(op, 1))) {
    if (intent(op) == "matrix")
      return op.emitError("engine=matrix does not yet support a reduction-first array snapshot; use bounded register tiles or express reduction-last snapshots"), failure();
    return ContractionEngine::Vector;
  }
  bool integer = integerMatrixGeometry(op, target, left, right);
  if (integer && arrayResult) {
    int64_t outputs = std::accumulate(output.begin(), output.end(), int64_t(1),
                                     std::multiplies<int64_t>());
    integer = storage.leftArray && element(op.getOperand(0).getType()).isInteger(8) &&
        ((storage.rightArray && element(op.getOperand(1).getType()).isInteger(8)) ||
         (storage.rightVector && outputs <= 4096));
  }
  if (integer && arrayResult && intent(op) == "matrix")
    return op.emitError("engine=matrix requires a bounded register tile; this array contraction has a runtime validity branch that can require vector execution"), failure();
  if (integer) return ContractionEngine::MatrixInteger;
  bool bfloat = arrayResult && target.hasFeature("amx-bf16") && matrixAxes(op) &&
      element(op.getResult(0).getType()).isF32() &&
      element(op.getOperand(0).getType()).isBF16() && element(op.getOperand(1).getType()).isBF16() &&
      storage.leftArray && storage.rightArray;
  if (bfloat) {
    bfloat = tileExtent(left[0], 1) && tileExtent(right[0], 1) &&
             left[1] >= 32 && left[1] % 32 == 0 && left[1] == right[1];
  }
  if (intent(op) == "matrix" && bfloat)
    return op.emitError("engine=matrix cannot use the guarded BF16 route: its full numerical contract may require vector execution for runtime input values"), failure();
  if (bfloat) return ContractionEngine::MatrixBFloat;
  if (intent(op) == "matrix")
    return op.emitError("engine=matrix has no legal x86 matrix implementation for these types, axes, extents, storage and target features"), failure();
  return ContractionEngine::Vector;
}

void weft::x86::recordContractionEngine(ModuleOp module, Operation &source, ContractionEngine engine, bool arrayResult) {
  Builder b(module.getContext());
  SmallVector<Attribute> records;
  if (auto existing = module->getAttrOfType<ArrayAttr>("weft.x86.contractions"))
    records.append(existing.begin(), existing.end());
  std::string location;
  llvm::raw_string_ostream stream(location);
  source.getLoc().print(stream);
  records.push_back(b.getDictionaryAttr({
      b.getNamedAttr("source", b.getStringAttr(location)),
      b.getNamedAttr("requested_engine", b.getStringAttr(intent(source))),
      b.getNamedAttr("selected_engine", b.getStringAttr(engine == ContractionEngine::Vector ? "vector" : "matrix")),
      b.getNamedAttr("instruction_family", b.getStringAttr(engine == ContractionEngine::MatrixInteger ? "amx-int8" :
          engine == ContractionEngine::MatrixBFloat ? "amx-bf16" : "simd")),
      b.getNamedAttr("conditional", b.getBoolAttr(arrayResult && engine != ContractionEngine::Vector)),
      b.getNamedAttr("selection_rule", b.getStringAttr("author constraint; legal matrix first; no cross-engine cost ranking"))}));
  module->setAttr("weft.x86.contractions", b.getArrayAttr(records));
}
