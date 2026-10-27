#include "Verification.h"

using namespace weft::kernel;
using namespace weft::kernel::detail;

mlir::LogicalResult ConstantOp::verify() {
  if (!isScalar(getResult().getType()) || getResult().getType().isIndex())
    return emitOpError(
        "constant represents a signed or unsigned scalar numeric literal");
  auto typed = mlir::dyn_cast<mlir::TypedAttr>(getValue());
  if (!typed || typed.getType() != getResult().getType())
    return emitOpError("constant attribute and result type must match");
  return mlir::success();
}

mlir::LogicalResult IotaOp::verify() {
  auto result = getResult().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  if ((!element && !result.getElementType().isIndex()) ||
      (element && element.isSigned()) || result.getShape().size() != 1 ||
      result.getAxisIds().size() != 1)
    return emitOpError("iota produces one unsigned integer or index logical axis");
  if (getEnd() <= getStart() || result.getShape()[0] != getEnd() - getStart())
    return emitOpError("iota shape must equal its positive [start, end) extent");
  return mlir::success();
}

mlir::LogicalResult ValueFactOp::verify() {
  if (getKind() != "divisibility" && getKind() != "contiguity" && getKind() != "constancy")
    return emitOpError("unknown logical-axis fact");
  auto inputs = getInputs();
  if (inputs.size() != 1 && inputs.size() != 2)
    return emitOpError("value facts require a numeric value or a pointer origin and offsets");
  mlir::Type numeric = inputs.back().getType();
  if (!isNumeric(numeric) || (inputs.size() == 2 &&
      (!mlir::isa<PointerType>(inputs.front().getType()) || !logicalElement(numeric).isInteger(64))))
    return emitOpError("value facts require a numeric domain and explicit pointer provenance");
  size_t rank = logicalShape(numeric).size();
  if (getKind() == "divisibility") rank = std::max(size_t{1}, rank);
  if (getValues().size() != rank || llvm::any_of(getValues(), [](int64_t n) { return n <= 0; }))
    return emitOpError("value facts require one positive integer per logical axis");
  return mlir::success();
}

mlir::LogicalResult NewOp::verify() {
  if (!isLocalValue(getResult().getType()))
    return emitOpError("new produces a local Value");
  if (getInitialized() != static_cast<bool>(getInitial()))
    return emitOpError("initialized must agree with the optional initial value");
  if (!getInitial())
    return emitOpError(
        "new requires an explicit initial value; uninitialized state has no canonical semantics");
  mlir::Type initial = getInitial().getType();
  mlir::Type result = getResult().getType();
  if (initial == result)
    return mlir::success();
  if (!logicalShape(initial).empty() || logicalElement(initial) != logicalElement(result))
    return emitOpError("new initial value must match or scalar-broadcast to the result");
  return mlir::success();
}

mlir::LogicalResult LocalShapeOp::verify() {
  auto owner = (*this)->getParentOfType<KernelOp>();
  if (!owner || getId() <= static_cast<int64_t>(owner.getShapeSymbols().size()))
    return emitOpError("local shape identity must be distinct from every public ABI shape");
  if (auto constant = getExtent().getDefiningOp<mlir::arith::ConstantIndexOp>();
      constant && constant.value() < 0)
    return emitOpError("local storage extent cannot be negative");
  return mlir::success();
}

void LocalShapeOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), LocalShapeResource::get());
}

mlir::LogicalResult EmptyOp::verify() {
  auto type = getResult().getType();
  if (!isNumeric(type) || type.getShape().size() != getExtents().size())
    return emitOpError("empty storage requires one explicit extent per numeric value axis");
  auto owner = (*this)->getParentOfType<KernelOp>();
  if (!owner) return emitOpError("empty storage must be task-local");
  for (auto [shape, extent] : llvm::zip(type.getShape().asArrayRef(), getExtents())) {
    if (auto constant = extent.getDefiningOp<mlir::arith::ConstantIndexOp>();
        constant && (constant.value() < 0 || (shape > 0 && constant.value() != shape)))
      return emitOpError("empty storage extent disagrees with its declared shape");
    if (shape < -static_cast<int64_t>(owner.getShapeSymbols().size()) &&
        weft::localShapeExtent<LocalShapeOp>(*this, -shape) != extent)
      return emitOpError("local shape must be bound by the dominating actual extent SSA");
  }
  return mlir::success();
}

mlir::LogicalResult MaterializeOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("materialize preserves its logical value type");
  return mlir::success();
}

mlir::LogicalResult BitcastOp::verify() {
  auto sourceWidth = bitWidth(getInput().getType());
  auto targetWidth = bitWidth(getResult().getType());
  if (!isLocalValue(getInput().getType()) || !isLocalValue(getResult().getType()) ||
      !isNumeric(getInput().getType()) || !isNumeric(getResult().getType()) ||
      !sourceWidth || !targetWidth || *sourceWidth != *targetWidth ||
      *sourceWidth < 8 || !sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError("bitcast preserves each numeric element's bit width and logical domain");
  return mlir::success();
}

mlir::LogicalResult UnaryOp::verify() {
  if (getInput().getType() != getResult().getType() ||
      !isNumeric(getInput().getType()))
    return emitOpError("unary operation preserves its logical type");
  if (getKind() != "neg" && getKind() != "abs" && getKind() != "exp" && getKind() != "exp2" &&
      getKind() != "sqrt" && getKind() != "rsqrt" && getKind() != "log")
    return emitOpError("unknown unary numeric operation");
  if ((getKind() == "exp" || getKind() == "exp2" || getKind() == "sqrt" ||
       getKind() == "rsqrt" || getKind() == "log") &&
      !mlir::isa<mlir::FloatType>(logicalElement(getInput().getType())))
    return emitOpError() << getKind() << " requires floating-point elements";
  return mlir::success();
}

mlir::LogicalResult BinaryOp::verify() {
  constexpr llvm::StringLiteral allowed[] = {
      "add", "sub", "mul", "div", "mod", "and", "or",
      "xor", "shl", "shr", "max", "min", "maximum", "minimum"};
  if (!llvm::is_contained(allowed, getKind()))
    return emitOpError("unknown binary numeric operation");
  if (failed(verifyPointwise(*this, getLhs().getType(), getRhs().getType(),
                             getResult().getType())))
    return mlir::failure();
  if ((getKind() == "maximum" || getKind() == "minimum") &&
      !mlir::isa<mlir::FloatType>(logicalElement(getResult().getType())))
    return emitOpError("propagating maximum/minimum requires floating elements");
  if (getKind() == "mod" || getKind() == "and" || getKind() == "or" ||
      getKind() == "xor" || getKind() == "shl" || getKind() == "shr") {
    mlir::Type element = logicalElement(getResult().getType());
    if (!element.isIndex() && !mlir::isa<mlir::IntegerType>(element))
      return emitOpError("integer-only binary operation has non-integer elements");
  }
  return mlir::success();
}

mlir::LogicalResult CompareOp::verify() {
  constexpr llvm::StringLiteral predicates[] = {"eq", "ne", "lt",
                                                 "le", "gt", "ge"};
  if (!llvm::is_contained(predicates, getPredicate()))
    return emitOpError("unknown comparison predicate");
  if (!isLocalValue(getLhs().getType()) || !isLocalValue(getRhs().getType()) ||
      !isNumeric(getLhs().getType()) || !isNumeric(getRhs().getType()) ||
      logicalElement(getLhs().getType()) != logicalElement(getRhs().getType()))
    return emitOpError("compare operands must have matching element types");
  auto predicate = mlir::dyn_cast<mlir::IntegerType>(logicalElement(getResult().getType()));
  if (!predicate || predicate.getWidth() != 1)
    return emitOpError("compare result must have i1 elements");
  llvm::SmallVector<int64_t> expected(logicalAxes(getLhs().getType()));
  for (int64_t axis : logicalAxes(getRhs().getType()))
    if (!llvm::is_contained(expected, axis))
      expected.push_back(axis);
  if (!llvm::equal(expected, logicalAxes(getResult().getType())))
    return emitOpError(
        "compare result axes must follow lhs order then rhs-only axes");
  for (auto [axis, extent] :
       llvm::zip(logicalAxes(getResult().getType()),
                 logicalShape(getResult().getType()))) {
    std::optional<int64_t> expectedExtent;
    for (mlir::Type operand : {getLhs().getType(), getRhs().getType()}) {
      auto operandAxes = logicalAxes(operand);
      auto found = llvm::find(operandAxes, axis);
      if (found == operandAxes.end())
        continue;
      int64_t operandExtent =
          logicalShape(operand)[static_cast<size_t>(found - operandAxes.begin())];
      if (!expectedExtent || *expectedExtent == 1)
        expectedExtent = operandExtent;
      else if (operandExtent != 1 && operandExtent != *expectedExtent)
        return emitOpError(
            "compare operands disagree on a shared logical axis extent");
    }
    if (!expectedExtent || extent != *expectedExtent)
      return emitOpError(
          "compare result extent must equal its broadcast operand extent");
  }
  return mlir::success();
}

mlir::LogicalResult SelectOp::verify() {
  mlir::Type result = getResult().getType();
  if (!isLocalValue(getCondition().getType()) ||
      !logicalElement(getCondition().getType()).isInteger(1) ||
      !isNumeric(getTrueValue().getType()) || !isNumeric(getFalseValue().getType()) ||
      !isLocalValue(result) ||
      logicalElement(getTrueValue().getType()) != logicalElement(result) ||
      logicalElement(getFalseValue().getType()) != logicalElement(result))
    return emitOpError("select requires an i1 predicate and matching numeric elements");
  llvm::SmallVector<int64_t> axes, shape;
  for (mlir::Value value : {getTrueValue(), getFalseValue(), getCondition()}) {
    if (!isLocalValue(value.getType())) return emitOpError("select operands must be local Values");
    for (auto [axis, extent] : llvm::zip(logicalAxes(value.getType()), logicalShape(value.getType()))) {
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) { axes.push_back(axis); shape.push_back(extent); }
      else {
        int64_t &previous = shape[found - axes.begin()];
        if (previous == 1) previous = extent;
        else if (extent != 1 && extent != previous)
          return emitOpError("select operands disagree on a shared axis extent");
      }
    }
  }
  if (!llvm::equal(axes, logicalAxes(result)) || !llvm::equal(shape, logicalShape(result)))
    return emitOpError("select result must preserve true/false/predicate broadcast axes in order");
  return mlir::success();
}

mlir::LogicalResult CastOp::verify() {
  if (!isNumeric(getInput().getType()) || !isNumeric(getResult().getType()) ||
      !sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError("cast converts numeric elements and preserves the logical domain");
  if (logicalElement(getInput().getType()) ==
      logicalElement(getResult().getType()))
    return emitOpError("identity conversion is not a canonical cast");
  return mlir::success();
}

mlir::LogicalResult NarrowOp::verify() {
  if (!isNumeric(getInput().getType()) || !isNumeric(getResult().getType()) ||
      !sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError(
        "narrow converts numeric elements and preserves the logical domain");
  if (getRounding() != "rne" && getRounding() != "rtz" &&
      getRounding() != "rdn" && getRounding() != "rup" &&
      getRounding() != "dynamic")
    return emitOpError("unknown numeric narrowing rounding mode");
  auto inputWidth = bitWidth(getInput().getType());
  auto resultWidth = bitWidth(getResult().getType());
  if (!inputWidth || !resultWidth || *resultWidth >= *inputWidth)
    return emitOpError("narrow must reduce the element bit width");
  if (getSaturate() &&
      !mlir::isa<mlir::IntegerType>(logicalElement(getResult().getType())))
    return emitOpError("saturation is defined only for an integer result");
  return mlir::success();
}

mlir::LogicalResult MacGroupsOp::verify() {
  if (getGroup() <= 0 || getOverflow() != "wrap")
    return emitOpError("mac_groups requires positive group and wrap overflow");
  if (!mlir::isa<ValueType>(getLhs().getType()) ||
      !mlir::isa<ValueType>(getRhs().getType()) ||
      !mlir::isa<ValueType>(getResult().getType()) ||
      !isNumeric(getLhs().getType()) || !isNumeric(getRhs().getType()) ||
      !isNumeric(getResult().getType()))
    return emitOpError("mac_groups consumes and produces shaped numeric Values");
  auto lhs = mlir::cast<ValueType>(getLhs().getType());
  auto rhs = mlir::cast<ValueType>(getRhs().getType());
  auto result = mlir::cast<ValueType>(getResult().getType());
  auto lhsAxes = lhs.getAxisIds().asArrayRef();
  auto rhsAxes = rhs.getAxisIds().asArrayRef();
  if (lhsAxes.empty() || rhsAxes.empty() || lhsAxes.back() != rhsAxes.back())
    return emitOpError("mac_groups requires one final common grouped axis");
  int64_t groupedAxis = lhsAxes.back();
  llvm::SmallVector<int64_t> inputAxes(lhsAxes.drop_back());
  llvm::SmallVector<int64_t> inputShape(
      lhs.getShape().asArrayRef().drop_back());
  for (auto [axis, extent] : llvm::zip(rhsAxes.drop_back(),
                                      rhs.getShape().asArrayRef().drop_back())) {
    auto found = llvm::find(inputAxes, axis);
    if (found == inputAxes.end()) {
      inputAxes.push_back(axis);
      inputShape.push_back(extent);
      continue;
    }
    size_t position = static_cast<size_t>(found - inputAxes.begin());
    if (inputShape[position] != extent)
      return emitOpError(
          "mac_groups operands disagree on a shared logical axis extent");
  }
  inputAxes.push_back(groupedAxis);
  inputShape.push_back(lhs.getShape()[lhs.getShape().size() - 1]);
  if (inputShape.back() != rhs.getShape()[rhs.getShape().size() - 1])
    return emitOpError(
        "mac_groups operands disagree on the grouped axis extent");
  if (inputAxes.empty() ||
      !llvm::equal(result.getAxisIds().asArrayRef(), inputAxes) ||
      result.getShape().size() != inputShape.size())
    return emitOpError(
        "mac_groups result axes must follow lhs order then rhs-only axes");
  for (size_t index = 0; index + 1 < inputShape.size(); ++index)
    if (result.getShape()[index] != inputShape[index])
      return emitOpError("mac_groups preserves non-grouped extents");
  if (inputShape.back() <= 0 || inputShape.back() % getGroup() ||
      result.getShape()[result.getShape().size() - 1] !=
          inputShape.back() / getGroup())
    return emitOpError("mac_groups divides its final logical extent by group");
  return mlir::success();
}

mlir::LogicalResult WidenOp::verify() {
  if (!sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError("widen preserves the logical domain");
  auto inputWidth = bitWidth(getInput().getType());
  auto resultWidth = bitWidth(getResult().getType());
  mlir::Type inputElement = logicalElement(getInput().getType());
  mlir::Type resultElement = logicalElement(getResult().getType());
  const bool integerToFloat = mlir::isa<mlir::IntegerType>(inputElement) &&
                              mlir::isa<mlir::FloatType>(resultElement);
  if (!inputWidth || !resultWidth ||
      (integerToFloat ? *resultWidth < *inputWidth
                      : *resultWidth <= *inputWidth))
    return emitOpError(
        "widen must increase width or convert an integer to no-narrower float");
  return mlir::success();
}

mlir::LogicalResult ReduceOp::verify() {
  if (getKind() != "add" && getKind() != "max" && getKind() != "min")
    return emitOpError("reduce kind must be add, max, or min");
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      !isLocalValue(getResult().getType()) || !isNumeric(getInput().getType()) ||
      !isNumeric(getResult().getType()))
    return emitOpError("reduce consumes and produces numeric Values");
  auto input = mlir::cast<ValueType>(getInput().getType());
  auto inputShape = input.getShape().asArrayRef();
  auto inputAxes = input.getAxisIds().asArrayRef();
  auto resultShape = logicalShape(getResult().getType());
  auto resultAxes = logicalAxes(getResult().getType());
  if (getAxis() < 0 || static_cast<size_t>(getAxis()) >= inputShape.size() ||
      resultShape.size() + 1 != inputShape.size() ||
      logicalElement(getResult().getType()) != input.getElementType())
    return emitOpError("reduce result must remove exactly its selected axis");
  size_t resultIndex = 0;
  for (size_t inputIndex = 0; inputIndex < inputShape.size(); ++inputIndex) {
    if (inputIndex == static_cast<size_t>(getAxis()))
      continue;
    if (resultShape[resultIndex] != inputShape[inputIndex] ||
        resultAxes[resultIndex] != inputAxes[inputIndex])
      return emitOpError("reduce must preserve every non-reduced logical axis");
    ++resultIndex;
  }
  return mlir::success();
}

mlir::LogicalResult Fold2Op::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      !mlir::isa<ValueType>(getResult().getType()))
    return emitOpError("fold2 consumes and produces shaped Values");
  auto input = mlir::cast<ValueType>(getInput().getType());
  auto result = mlir::cast<ValueType>(getResult().getType());
  auto inputElement = mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  llvm::ArrayRef<int64_t> inputShape = input.getShape().asArrayRef();
  llvm::ArrayRef<int64_t> resultShape = result.getShape().asArrayRef();
  if (inputShape.empty() || inputShape.size() != resultShape.size() ||
      inputShape.back() <= 0 || inputShape.back() % 2 ||
      resultShape.back() != inputShape.back() / 2 ||
      !inputElement || !resultElement || !resultElement.isSigned() ||
      resultElement.getWidth() != 32 ||
      input.getAxisIds() != result.getAxisIds() ||
      !llvm::equal(inputShape.drop_back(), resultShape.drop_back()))
    return emitOpError("fold2 halves an even final logical extent");
  return mlir::success();
}

mlir::LogicalResult ReduceRegionOp::verify() {
  return weft::verifyReductionRegion<ValueType, ReduceYieldOp>(*this, getAxis(), getCombine(),
      [](mlir::Operation *op) {
        return mlir::isa<ConstantOp, mlir::arith::ConstantOp, UnaryOp, BinaryOp,
            CompareOp, SelectOp, CastOp, BitcastOp, WidenOp, NarrowOp>(op);
      });
}

mlir::LogicalResult ScanOp::verify() {
  return weft::verifyReductionRegion<ValueType, ScanYieldOp>(*this, getAxis(), getCombine(),
      [](mlir::Operation *op) {
        return mlir::isa<ConstantOp, mlir::arith::ConstantOp, UnaryOp, BinaryOp,
            CompareOp, SelectOp, CastOp, BitcastOp, WidenOp, NarrowOp>(op);
      }, true);
}

mlir::LogicalResult DotOp::verify() {
  if (getOver().size() != 1 || logicalAxes(getLhs().getType()).empty() ||
      logicalAxes(getRhs().getType()).empty() ||
      getOver()[0] != logicalAxes(getLhs().getType()).back() ||
      getOver()[0] != logicalAxes(getRhs().getType()).back())
    return emitOpError("dot reduces the final common logical axis");
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult ContractOp::verify() {
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult OuterContractOp::verify() {
  llvm::DenseSet<int64_t> reduction;
  for (int64_t axis : getOver())
    reduction.insert(axis);
  for (int64_t axis : logicalAxes(getLhs().getType()))
    if (!reduction.contains(axis) &&
        llvm::is_contained(logicalAxes(getRhs().getType()), axis))
      return emitOpError(
          "outer_contract requires disjoint lhs and rhs free axes");
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult LookupOp::verify() {
  if (getBounds() != "in_bounds")
    return emitOpError("current canonical lookup requires bounds = in_bounds");
  mlir::Type indexElement = logicalElement(getIndices().getType());
  auto index = mlir::dyn_cast<mlir::IntegerType>(indexElement);
  if (!mlir::isa<ValueType>(getTable().getType()) ||
      !isNumeric(getTable().getType()) ||
      (!indexElement.isIndex() && (!index || index.isSigned())))
    return emitOpError(
        "lookup requires a numeric local table and unsigned integer indices");
  if (!sameDomain(getIndices().getType(), getResult().getType()))
    return emitOpError("lookup result must follow the index domain");
  if (logicalElement(getTable().getType()) != logicalElement(getResult().getType()))
    return emitOpError("lookup result element type must match the table element type");
  return mlir::success();
}
