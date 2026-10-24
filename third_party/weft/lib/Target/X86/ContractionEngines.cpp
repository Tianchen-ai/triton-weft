#include "ContractionEngines.h"
#include "Weft/Analysis/LogicalAxes.h"
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
  return wideInputs <= 1 && tileExtent(lhs[0], 1) && tileExtent(rhs[0], 1) &&
         lhs[1] >= 16 && lhs[1] % 16 == 0 && lhs[1] == rhs[1];
}

FailureOr<weft::x86::ContractionEngine> weft::x86::selectContractionEngine(
    Operation &op, const X86Target &target, ContractionStorage storage, bool arrayResult,
    ArrayRef<int64_t> left, ArrayRef<int64_t> right, ArrayRef<int64_t> output) {
  if (intent(op) == "vector") return ContractionEngine::Vector;
  bool integer = integerMatrixGeometry(op, target, left, right);
  if (integer && arrayResult) {
    int64_t outputs = std::accumulate(output.begin(), output.end(), int64_t(1),
                                     std::multiplies<int64_t>());
    integer = storage.leftArray && element(op.getOperand(0).getType()).isInteger(8) &&
        ((storage.rightArray && element(op.getOperand(1).getType()).isInteger(8)) ||
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
  if (intent(op) == "matrix" && bfloat)
    return op.emitError("engine=matrix cannot use the guarded BF16 route: its full numerical contract may require vector execution for runtime input values"), failure();
  if (bfloat) return ContractionEngine::MatrixBFloat;
  if (intent(op) == "matrix")
    return op.emitError("engine=matrix has no legal x86 matrix implementation for these types, axes, extents, storage and target features"), failure();
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
  records.push_back(b.getDictionaryAttr({
      b.getNamedAttr("source", b.getStringAttr(location)),
      b.getNamedAttr("requested_engine", b.getStringAttr(intent(source))),
      b.getNamedAttr("selected_engine", b.getStringAttr(engine == ContractionEngine::Vector ? "vector" : "matrix")),
      b.getNamedAttr("instruction_family", b.getStringAttr(engine == ContractionEngine::MatrixInteger ? "amx-int8" :
          engine == ContractionEngine::MatrixBFloat ? "amx-bf16" : "simd")),
      b.getNamedAttr("conditional", b.getBoolAttr(engine == ContractionEngine::MatrixBFloat)),
      b.getNamedAttr("selection_rule", b.getStringAttr("author constraint; legal matrix first; no cross-engine cost ranking"))}));
  module->setAttr("weft.x86.contractions", b.getArrayAttr(records));
}
