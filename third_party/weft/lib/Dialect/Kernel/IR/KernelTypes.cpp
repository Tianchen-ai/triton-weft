#include "Verification.h"

using namespace weft::kernel;
using namespace weft::kernel::detail;

mlir::LogicalResult EncodingType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef family, llvm::StringRef kind,
    llvm::StringRef layoutIdentity, mlir::DenseI64ArrayAttr parameters) {
  if (family.empty())
    return emitError() << "encoding family must not be empty";
  if (kind != "base" && kind != "dense" && kind != "derived_family" &&
      kind != "derived_instance")
    return emitError() << "unknown encoding kind";
  if ((kind == "base" || kind == "dense" || kind == "derived_instance") &&
      layoutIdentity.empty())
    return emitError() << "concrete encoding requires a layout identity";
  if (kind == "derived_family" && !layoutIdentity.empty())
    return emitError() << "abstract derived encoding cannot carry a layout identity";
  if ((kind == "base" || kind == "dense" || kind == "derived_family") &&
      !parameters.empty())
    return emitError() << "only a concrete derived encoding carries static parameters";
  if (kind == "derived_instance") {
    if (parameters.empty())
      return emitError() << "derived instance requires static parameters";
    if (llvm::any_of(parameters.asArrayRef(),
                     [](int64_t value) { return value <= 0; }))
      return emitError() << "derived encoding parameters must be positive";
  }
  return mlir::success();
}

mlir::LogicalResult PointerType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type elementType) {
  if (!isCanonicalNumericElement(elementType) || elementType.isIndex())
    return emitError() << "pointer element must be a storage scalar";
  unsigned bits = elementType.getIntOrFloatBitWidth();
  if (bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return emitError() << "pointer element must have a native storage width";
  return mlir::success();
}

mlir::LogicalResult ViewType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type encoding, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isEncoding(encoding))
    return emitError() << "View encoding must be an Encoding type";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true);
}

mlir::LogicalResult ValueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isCanonicalNumericElement(elementType) && !isEncoding(elementType))
    return emitError() << "Value element must be numeric or encoded";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(),
                     isCanonicalNumericElement(elementType));
}

mlir::LogicalResult SliceType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type encoding, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isEncoding(encoding))
    return emitError() << "slice encoding must be an Encoding type";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true);
}

mlir::LogicalResult DomainType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef axisName, int64_t domainId, int64_t parentDomainId,
    int64_t axisId, llvm::StringRef relation, llvm::StringRef tail) {
  if (axisName.empty())
    return emitError() << "domain axis name must not be empty";
  if (relation != "root" && relation != "rows" && relation != "cols" &&
      relation != "tiles" && relation != "blocks" && relation != "subs")
    return emitError() << "unknown level relation";
  if (relation == "root") {
    if (domainId != 0 || parentDomainId != -1 || axisId != 0)
      return emitError()
             << "root domain has id zero, no parent, and axis identity zero";
  } else if (domainId <= 0 || parentDomainId < 0 || axisId <= 0) {
    return emitError()
           << "child domain has positive domain/axis identities and a parent";
  }
  if (tail != "exact" && tail != "tail")
    return emitError() << "domain tail must be exact or tail";
  return mlir::success();
}

mlir::LogicalResult PointType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    DomainType domain) {
  if (!domain)
    return emitError() << "point must reference a domain";
  return mlir::success();
}
