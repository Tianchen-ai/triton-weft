#include "Verification.h"

using namespace weft::kernel;

namespace weft::kernel::detail {
bool isScalar(mlir::Type type) {
  return type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

bool isEncoding(mlir::Type type) { return mlir::isa<EncodingType>(type); }

bool isViewLike(mlir::Type type) {
  return mlir::isa<ViewType, SliceType>(type);
}

bool isLocalValue(mlir::Type type) {
  return isScalar(type) || isEncoding(type) || mlir::isa<ValueType>(type);
}

bool isCanonicalNumericElement(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth() == 1 || !integer.isSignless();
  return type.isIndex() || mlir::isa<mlir::FloatType>(type);
}

bool isCanonicalProgramType(mlir::Type type) {
  if (isCanonicalNumericElement(type))
    return true;
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return isEncoding(value.getElementType()) ||
           isCanonicalNumericElement(value.getElementType());
  return mlir::isa<EncodingType, ViewType, SliceType, PointerType, DomainType, PointType>(type);
}

bool isNumeric(mlir::Type type) {
  type = logicalElement(type);
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type) || type.isIndex();
}

std::optional<unsigned> bitWidth(mlir::Type type) {
  type = logicalElement(type);
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return std::nullopt;
}

std::string denseFamilyFor(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    llvm::StringRef prefix = integer.isUnsigned() ? "u" : "i";
    return (prefix + llvm::Twine(integer.getWidth())).str();
  }
  if (type.isBF16())
    return "bf16";
  if (mlir::isa<mlir::Float8E4M3FNType>(type)) return "f8E4M3FN";
  if (mlir::isa<mlir::Float8E5M2Type>(type)) return "f8E5M2";
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return ("f" + llvm::Twine(floating.getWidth())).str();
  return {};
}

EncodingDeclOp findEncodingDeclaration(mlir::Operation *operation,
                                       llvm::StringRef family) {
  EncodingDeclOp result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](EncodingDeclOp declaration) {
      if (!result && declaration.getSymName() == family)
        result = declaration;
    });
  return result;
}

mlir::LogicalResult verifyEncodingReference(mlir::Operation *operation,
                                            EncodingType encoding) {
  if (encoding.getKind() == "dense") {
    if (encoding.getLayoutIdentity() !=
        ("dense." + encoding.getFamily()).str()) {
      operation->emitOpError()
          << "dense Encoding layout identity must be dense.<family>";
      return mlir::failure();
    }
    return mlir::success();
  }

  auto module = operation->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    operation->emitOpError("Encoding reference must belong to one module");
    return mlir::failure();
  }
  if (encoding.getKind() == "base") {
    unsigned matches = 0;
    module.walk([&](EncodingDeclOp declaration) {
      if (declaration.getSymName() == encoding.getFamily() &&
          declaration.getLayoutIdentity() == encoding.getLayoutIdentity())
        ++matches;
    });
    if (matches != 1) {
      operation->emitOpError()
          << "base Encoding must resolve to exactly one declaration with the "
             "same family and pinned layout identity";
      return mlir::failure();
    }
    return mlir::success();
  }
  if (encoding.getKind() == "derived_instance") {
    unsigned matches = 0;
    module.walk([&](DeriveOp derive) {
      if (derive.getResultFamily() == encoding.getFamily() &&
          derive.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derive.getParameterValues() == encoding.getParameters().asArrayRef())
        ++matches;
    });
    if (matches != 1) {
      operation->emitOpError()
          << "derived Encoding instance must resolve to exactly one derive "
             "builder with the same family, parameters, and pinned identity";
      return mlir::failure();
    }
    return mlir::success();
  }
  operation->emitOpError(
      "abstract derived Encoding family cannot appear in a runtime program type");
  return mlir::failure();
}

mlir::LogicalResult verifyEncodingInType(mlir::Operation *operation,
                                         mlir::Type type) {
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return verifyEncodingReference(
        operation, mlir::cast<EncodingType>(view.getEncoding()));
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return verifyEncodingReference(
        operation, mlir::cast<EncodingType>(slice.getEncoding()));
  if (auto value = mlir::dyn_cast<ValueType>(type)) {
    if (auto encoding = mlir::dyn_cast<EncodingType>(value.getElementType()))
      return verifyEncodingReference(operation, encoding);
    return mlir::success();
  }
  if (auto encoding = mlir::dyn_cast<EncodingType>(type))
    return verifyEncodingReference(operation, encoding);
  return mlir::success();
}

llvm::StringRef baseFamilyFor(mlir::Operation *operation,
                              EncodingType encoding) {
  if (encoding.getKind() == "base")
    return encoding.getFamily();
  if (encoding.getKind() != "derived_instance")
    return {};
  llvm::StringRef source;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](DeriveOp derive) {
      if (source.empty() && derive.getResultFamily() == encoding.getFamily() &&
          derive.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derive.getParameterValues() ==
              encoding.getParameters().asArrayRef())
        source = derive.getSourceFamily();
    });
  return source;
}

mlir::LogicalResult verifyShape(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> axes,
    bool allowEmpty) {
  if (!allowEmpty && shape.empty())
    return emitError() << "logical Value must have positive rank";
  if (shape.size() != axes.size())
    return emitError() << "logical shape and axis identity rank must match";
  llvm::DenseSet<int64_t> seen;
  for (auto [extent, axis] : llvm::zip(shape, axes)) {
    if (extent == 0)
      return emitError() << "logical extents must be positive or symbolic";
    if (axis <= 0)
      return emitError() << "logical axis identities must be positive";
    if (!seen.insert(axis).second)
      return emitError() << "one logical axis identity cannot occur twice";
  }
  return mlir::success();
}

mlir::LogicalResult verifyStringArray(mlir::Operation *operation,
                                      mlir::ArrayAttr values,
                                      llvm::StringRef name) {
  if (llvm::any_of(values, [](mlir::Attribute value) {
        return !mlir::isa<mlir::StringAttr>(value);
      }))
    return operation->emitOpError() << name << " must contain only strings";
  return mlir::success();
}

mlir::LogicalResult verifyPointwise(mlir::Operation *operation,
                                    mlir::Type lhs, mlir::Type rhs,
                                    mlir::Type result) {
  if (!isLocalValue(lhs) || !isLocalValue(rhs) || !isLocalValue(result))
    return operation->emitOpError("pointwise operands must be local Values");
  if (!isNumeric(lhs) || !isNumeric(rhs) || !isNumeric(result))
    return operation->emitOpError("pointwise operands and result must be numeric");
  if (logicalElement(lhs) != logicalElement(rhs))
    return operation->emitOpError("pointwise element types must match");
  if (logicalElement(result) != logicalElement(lhs))
    return operation->emitOpError("pointwise result element type must match");
  llvm::SmallVector<int64_t> expectedAxes(logicalAxes(lhs));
  for (int64_t axis : logicalAxes(rhs))
    if (!llvm::is_contained(expectedAxes, axis))
      expectedAxes.push_back(axis);
  if (!llvm::equal(expectedAxes, logicalAxes(result)))
    return operation->emitOpError(
        "pointwise result axes must follow lhs order then rhs-only axes");
  for (auto [axis, extent] : llvm::zip(logicalAxes(result), logicalShape(result))) {
    std::optional<int64_t> expectedExtent;
    for (mlir::Type operand : {lhs, rhs}) {
      auto operandAxes = logicalAxes(operand);
      auto found = llvm::find(operandAxes, axis);
      if (found == operandAxes.end())
        continue;
      size_t position = static_cast<size_t>(found - operandAxes.begin());
      int64_t operandExtent = logicalShape(operand)[position];
      if (!expectedExtent || *expectedExtent == 1)
        expectedExtent = operandExtent;
      else if (operandExtent != 1 && operandExtent != *expectedExtent)
        return operation->emitOpError(
            "pointwise operands disagree on a shared logical axis extent");
    }
    if (!expectedExtent || extent != *expectedExtent)
      return operation->emitOpError(
          "pointwise result extent must equal its broadcast operand extent");
  }
  return mlir::success();
}

mlir::LogicalResult verifyProjection(mlir::Operation *operation,
                                     mlir::Type base,
                                     mlir::ValueRange indices,
                                     mlir::ArrayAttr selectors,
                                     mlir::Type projected,
                                     bool allowGather) {
  if (selectors.size() > logicalAxes(base).size())
    return operation->emitOpError("projection has more selectors than input axes");
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  size_t indexCursor = 0;
  constexpr int64_t dynamicProjection = std::numeric_limits<int64_t>::min();
  for (size_t position = 0; position < logicalAxes(base).size(); ++position) {
    llvm::StringRef selector =
        position < selectors.size()
            ? mlir::dyn_cast<mlir::StringAttr>(selectors[position]).getValue()
            : llvm::StringRef("all");
    if (selector == "all") {
      expectedAxes.push_back(logicalAxes(base)[position]);
      expectedShape.push_back(logicalShape(base)[position]);
      continue;
    }
    if (indexCursor >= indices.size())
      return operation->emitOpError(
          "projection selector has no corresponding index operand");
    mlir::Value index = indices[indexCursor++];
    if (selector == "domain" || selector == "group_index") {
      auto point = mlir::dyn_cast<PointType>(index.getType());
      if (!point || point.getDomain().getAxisId() != logicalAxes(base)[position])
        return operation->emitOpError(
            "level projection point must select the corresponding logical axis");
      if (selector == "domain") {
        expectedAxes.push_back(logicalAxes(base)[position]);
        expectedShape.push_back(dynamicProjection);
      }
      continue;
    }
    if (selector == "index") {
      if (!index.getType().isIndex())
        return operation->emitOpError("scalar projection index must have index type");
      continue;
    }
    if (selector != "gather" || !allowGather)
      return operation->emitOpError("unknown or disallowed projection selector");
    auto gather = mlir::dyn_cast<ValueType>(index.getType());
    auto integer = gather
                       ? mlir::dyn_cast<mlir::IntegerType>(gather.getElementType())
                       : mlir::IntegerType();
    if (!integer || integer.isSigned())
      return operation->emitOpError(
          "gather projection requires shaped unsigned indices");
    for (auto [axis, extent] : llvm::zip(gather.getAxisIds().asArrayRef(),
                                        gather.getShape().asArrayRef())) {
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return operation->emitOpError(
            "gather projection disagrees on a shared logical axis extent");
      }
    }
  }
  if (indexCursor != indices.size() ||
      !llvm::equal(expectedAxes, logicalAxes(projected)) ||
      expectedShape.size() != logicalShape(projected).size())
    return operation->emitOpError(
        "projection operands and result logical axes do not match");
  for (auto [expected, actual] : llvm::zip(expectedShape, logicalShape(projected)))
    if (expected != dynamicProjection && expected != actual)
      return operation->emitOpError(
          "projection result has an incorrect logical extent");
  return mlir::success();
}

mlir::Type promotedElement(mlir::Type lhs, mlir::Type rhs) {
  lhs = logicalElement(lhs);
  rhs = logicalElement(rhs);
  auto lhsFloat = mlir::dyn_cast<mlir::FloatType>(lhs);
  auto rhsFloat = mlir::dyn_cast<mlir::FloatType>(rhs);
  if (lhsFloat || rhsFloat) {
    unsigned width = std::max(lhsFloat ? lhsFloat.getWidth() : 0u,
                              rhsFloat ? rhsFloat.getWidth() : 0u);
    if (width > 32)
      return mlir::Float64Type::get(lhs.getContext());
    return mlir::Float32Type::get(lhs.getContext());
  }
  auto lhsInteger = mlir::dyn_cast<mlir::IntegerType>(lhs);
  auto rhsInteger = mlir::dyn_cast<mlir::IntegerType>(rhs);
  if (!lhsInteger || !rhsInteger || lhsInteger.isSignless() ||
      rhsInteger.isSignless())
    return {};
  unsigned width = std::max(lhsInteger.getWidth(), rhsInteger.getWidth());
  for (unsigned candidate : {8u, 16u, 32u, 64u})
    if (width <= candidate) {
      auto signedness = lhsInteger.isSigned() || rhsInteger.isSigned()
                            ? mlir::IntegerType::Signed
                            : mlir::IntegerType::Unsigned;
      return mlir::IntegerType::get(lhs.getContext(), candidate, signedness);
    }
  return {};
}

mlir::LogicalResult verifyContractLike(mlir::Operation *operation,
                                       mlir::Value lhs, mlir::Value rhs,
                                       llvm::ArrayRef<int64_t> over,
                                       std::optional<mlir::Type> accType,
                                       mlir::Type result) {
  if (auto intent = operation->getAttrOfType<mlir::StringAttr>("engine"))
    if (intent.getValue() != "auto" && intent.getValue() != "matrix" && intent.getValue() != "vector")
      return operation->emitOpError("engine must be auto, matrix, or vector");
  if (!isLocalValue(lhs.getType()) || !isLocalValue(rhs.getType()) ||
      !isLocalValue(result))
    return operation->emitOpError("contraction operands must be local Values");
  if (!isNumeric(lhs.getType()) || !isNumeric(rhs.getType()) || !isNumeric(result))
    return operation->emitOpError("contraction operands and result must be numeric");
  if (over.empty())
    return operation->emitOpError("contraction requires at least one reduction axis");
  llvm::DenseSet<int64_t> reduction;
  for (int64_t axis : over) {
    if (axis <= 0 || !llvm::is_contained(logicalAxes(lhs.getType()), axis) ||
        !llvm::is_contained(logicalAxes(rhs.getType()), axis) ||
        !reduction.insert(axis).second)
      return operation->emitOpError(
          "each contraction axis must occur once in both operands");
  }
  for (int64_t axis : logicalAxes(lhs.getType())) {
    auto rhsAxes = logicalAxes(rhs.getType());
    auto found = llvm::find(rhsAxes, axis);
    if (found == rhsAxes.end())
      continue;
    size_t lhsPosition = static_cast<size_t>(
        llvm::find(logicalAxes(lhs.getType()), axis) - logicalAxes(lhs.getType()).begin());
    size_t rhsPosition = static_cast<size_t>(found - rhsAxes.begin());
    if (logicalShape(lhs.getType())[lhsPosition] !=
        logicalShape(rhs.getType())[rhsPosition])
      return operation->emitOpError(
          "contraction operands disagree on a shared logical axis extent");
  }
  llvm::SmallVector<int64_t> expected;
  for (int64_t axis : logicalAxes(lhs.getType()))
    if (!reduction.contains(axis))
      expected.push_back(axis);
  for (int64_t axis : logicalAxes(rhs.getType()))
    if (!reduction.contains(axis) && !llvm::is_contained(expected, axis))
      expected.push_back(axis);
  if (!llvm::equal(expected, logicalAxes(result)))
    return operation->emitOpError(
        "contraction result axes must follow lhs free axes then rhs-only free axes");
  for (auto [axis, extent] : llvm::zip(logicalAxes(result), logicalShape(result))) {
    auto lhsAxes = logicalAxes(lhs.getType());
    auto lhsPosition = llvm::find(lhsAxes, axis);
    auto rhsAxes = logicalAxes(rhs.getType());
    auto rhsPosition = llvm::find(rhsAxes, axis);
    int64_t expectedExtent =
        lhsPosition != lhsAxes.end()
            ? logicalShape(lhs.getType())[static_cast<size_t>(lhsPosition -
                                                             lhsAxes.begin())]
            : logicalShape(rhs.getType())[static_cast<size_t>(rhsPosition -
                                                             rhsAxes.begin())];
    if (extent != expectedExtent)
      return operation->emitOpError(
          "contraction result extent must follow its free operand axis");
  }
  mlir::Type expectedElement =
      accType ? *accType : promotedElement(lhs.getType(), rhs.getType());
  if (!expectedElement || logicalElement(result) != expectedElement)
    return operation->emitOpError(
        "contraction result element must match acc_type or canonical promotion");
  return mlir::success();
}

} // namespace weft::kernel::detail
