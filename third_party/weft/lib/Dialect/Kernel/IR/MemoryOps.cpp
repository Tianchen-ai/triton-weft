#include "Verification.h"

using namespace weft::kernel;
using namespace weft::kernel::detail;

static mlir::LogicalResult verifyPointerMemory(mlir::Operation *op, mlir::Value base,
    mlir::Value offsets, mlir::Value mask, mlir::Value value) {
  auto element = mlir::cast<PointerType>(base.getType()).getElementType();
  if (logicalElement(value.getType()) != element ||
      !logicalElement(mask.getType()).isInteger(1) ||
      !mlir::isa<mlir::IntegerType>(logicalElement(offsets.getType())) ||
      !sameDomain(value.getType(), offsets.getType()) ||
      !sameDomain(value.getType(), mask.getType()))
    return op->emitOpError("pointer memory requires typed element offsets, i1 mask and matching value domains");
  return mlir::success();
}

mlir::LogicalResult PointerCastOp::verify() {
  auto bytes = [](PointerType type) { return std::max(8u, type.getElementType().getIntOrFloatBitWidth()) / 8; };
  if (bytes(getInput().getType()) != bytes(getResult().getType()))
    return emitOpError("pointer reinterpretation preserves storage element width");
  return mlir::success();
}

mlir::LogicalResult PointerLoadOp::verify() {
  if (getOther().getType() != getResult().getType())
    return emitOpError("masked load fill must have the result type");
  return verifyPointerMemory(*this, getBase(), getOffsets(), getMask(), getOther());
}

mlir::LogicalResult PointerStoreOp::verify() {
  return verifyPointerMemory(*this, getBase(), getOffsets(), getMask(), getValue());
}

mlir::LogicalResult PointerAtomicOp::verify() {
  auto element = logicalElement(getResult().getType());
  auto pointer = mlir::cast<PointerType>(getBase().getType());
  auto index = mlir::dyn_cast<mlir::IntegerType>(logicalElement(getOffsets().getType()));
  if (pointer.getElementType() != element || !index || !index.isSigned() || index.getWidth() != 64 ||
      !logicalElement(getMask().getType()).isInteger(1) || !sameDomain(getOffsets().getType(), getResult().getType()))
    return emitOpError("atomic pointer offset/result domain and element types must match");
  for (mlir::Value value : {getMask(), getValue(), getCompare()})
    if (!logicalShape(value.getType()).empty() && !sameDomain(value.getType(), getResult().getType()))
      return emitOpError("atomic operands must be scalar or share the pointer domain");
  if (logicalElement(getValue().getType()) != element || logicalElement(getCompare().getType()) != element)
    return emitOpError("atomic value and comparison preserve the storage dtype");
  return weft::verifyAtomicContract(*this, element, getKind(), getOrdering(), getScope());
}

mlir::LogicalResult AdmitOp::verify() {
  if (!isViewLike(getRegion().getType()) || !isLocalValue(getResult().getType()))
    return emitOpError("admit maps a View region to a local Value");
  mlir::Type resultElement = logicalElement(getResult().getType());
  if (auto encoded = mlir::dyn_cast<EncodingType>(resultElement)) {
    auto regionEncoding =
        mlir::cast<EncodingType>(logicalElement(getRegion().getType()));
    if (encoded != regionEncoding ||
        !sameDomain(getRegion().getType(), getResult().getType()))
      return emitOpError(
          "encoded admit preserves the selected logical record domain");
  } else if (!sameDomain(getRegion().getType(), getResult().getType())) {
    return emitOpError("dense admit must preserve the logical domain");
  }
  return mlir::success();
}

mlir::LogicalResult CommitOp::verify() {
  if (!isLocalValue(getValue().getType()) || !isViewLike(getRegion().getType()))
    return emitOpError("commit maps a local Value to a View region");
  if (!sameDomain(getValue().getType(), getRegion().getType()))
    return emitOpError("commit value and destination must have the same domain");
  auto destination =
      mlir::cast<EncodingType>(logicalElement(getRegion().getType()));
  if (destination.getKind() == "dense") {
    if (!isNumeric(getValue().getType()) ||
        destination.getFamily() != denseFamilyFor(logicalElement(getValue().getType())))
      return emitOpError(
          "dense commit requires an explicitly converted matching element type");
  } else if (logicalElement(getValue().getType()) != destination) {
    return emitOpError("encoded commit requires identical encoding");
  }
  return mlir::success();
}

mlir::LogicalResult AtomicRMWOp::verify() {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(getValue().getType());
  if (!integer || integer.isSignless() ||
      (integer.getWidth() != 32 && integer.getWidth() != 64) ||
      getResult().getType() != integer ||
      !logicalShape(getRegion().getType()).empty())
    return emitOpError("atomic update requires one scalar i32/u32/i64/u64 address and matching old-value result");
  auto encoding = mlir::cast<EncodingType>(logicalElement(getRegion().getType()));
  if (encoding.getKind() != "dense" ||
      encoding.getFamily() != denseFamilyFor(integer))
    return emitOpError("atomic update preserves the dense storage element type");
  if (getKind() != "add")
    return emitOpError("unsupported atomic update kind");
  if (getOrdering() != "relaxed" && getOrdering() != "acquire" &&
      getOrdering() != "release" && getOrdering() != "acq_rel" &&
      getOrdering() != "seq_cst")
    return emitOpError("unsupported atomic memory ordering");
  return mlir::success();
}

mlir::LogicalResult SliceOp::verify() {
  if (!isViewLike(getBase().getType()))
    return emitOpError("slice base must be a View or slice");
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return mlir::failure();
  if (logicalElement(getBase().getType()) != logicalElement(getResult().getType()))
    return emitOpError("slice preserves its View encoding");
  return verifyProjection(*this, getBase().getType(), getIndices(),
                          getSelectors(), getResult().getType(), false);
}

mlir::LogicalResult SubviewOp::verify() {
  if (!isViewLike(getBase().getType()))
    return emitOpError("subview base must be a View or slice");
  if (logicalElement(getBase().getType()) !=
          logicalElement(getResult().getType()) ||
      !llvm::equal(logicalAxes(getBase().getType()),
                   logicalAxes(getResult().getType())))
    return emitOpError("subview preserves Encoding and logical axes");
  return verifySubviewBounds(*this, logicalShape(getBase().getType()),
      logicalShape(getResult().getType()), getOffsetsAttr(), getExtentsAttr(),
      getDynamicBounds());
}

mlir::LogicalResult ExtentOp::verify() {
  if (!isViewLike(getBase().getType()) || getAxis() < 0 ||
      static_cast<size_t>(getAxis()) >= logicalShape(getBase().getType()).size())
    return emitOpError("extent requires a View or slice and an in-range axis position");
  return mlir::success();
}

mlir::LogicalResult ReshapeOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  if (input.getElementType() != result.getElementType())
    return emitOpError("reshape preserves the logical element type");
  auto inputCount = staticElementCount(input);
  auto resultCount = staticElementCount(result);
  bool axisRename = input.getShape() == result.getShape() &&
                    llvm::equal(getOrder(), input.getAxisIds().asArrayRef());
  bool unitProjection = isUnitAxisProjection(input.getShape(), input.getAxisIds(),
      result.getShape(), result.getAxisIds(), getOrder());
  if (!axisRename && !unitProjection && (!inputCount || !resultCount || *inputCount != *resultCount))
    return emitOpError(
        "reshape requires equal static element counts, an ordered shape rename, or a unit-axis projection");
  auto order = getOrder();
  auto inputAxes = input.getAxisIds().asArrayRef();
  if (order.size() != inputAxes.size())
    return emitOpError("reshape order must name every input logical axis");
  llvm::DenseSet<int64_t> seen;
  for (int64_t axis : order)
    if (!llvm::is_contained(inputAxes, axis) || !seen.insert(axis).second)
      return emitOpError(
          "reshape order must be a permutation of the input logical axes");
  return mlir::success();
}

mlir::LogicalResult FieldOp::verify() {
  mlir::Type owner = getOwner().getType();
  mlir::Type result = getResult().getType();
  if ((!isLocalValue(owner) && !isViewLike(owner)) ||
      !isEncoding(logicalElement(owner)) || getName().empty())
    return emitOpError(
        "field access requires a named field of an encoded Value or View region");
  auto ownerEncoding = mlir::cast<EncodingType>(logicalElement(owner));
  llvm::StringRef baseFamily = baseFamilyFor(*this, ownerEncoding);
  EncodingDeclOp declaration = findEncodingDeclaration(*this, baseFamily);
  if (!declaration)
    return emitOpError("encoded field owner has no matching base declaration");
  size_t fieldIndex = declaration.getFieldNames().size();
  for (auto [index, attribute] : llvm::enumerate(declaration.getFieldNames()))
    if (mlir::cast<mlir::StringAttr>(attribute).getValue() == getName()) {
      fieldIndex = index;
      break;
    }
  if (fieldIndex == declaration.getFieldNames().size())
    return emitOpError("field name is absent from the owner Encoding declaration");
  mlir::Type fieldType =
      mlir::cast<mlir::TypeAttr>(declaration.getFieldTypes()[fieldIndex]).getValue();
  auto fieldShape =
      mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[fieldIndex])
          .asArrayRef();
  auto ownerShape = logicalShape(owner);
  auto ownerAxes = logicalAxes(owner);
  auto resultShape = logicalShape(result);
  auto resultAxes = logicalAxes(result);
  if (ownerShape.empty() || ownerShape.back() != declaration.getElements())
    return emitOpError(
        "encoded field owner must expose exactly one complete record axis");
  llvm::SmallVector<int64_t> expectedShape(ownerShape.drop_back());
  expectedShape.append(fieldShape.begin(), fieldShape.end());
  if (resultAxes.size() < ownerAxes.size() - 1 ||
      (!fieldShape.empty() && resultAxes.empty()) ||
      !llvm::equal(resultShape, expectedShape) ||
      !llvm::equal(resultAxes.take_front(ownerAxes.size() - 1),
                   ownerAxes.drop_back()) ||
      (!fieldShape.empty() && resultAxes.back() != ownerAxes.back()))
    return emitOpError(
        "field result must replace the record axis with its declared field shape");
  if (isLocalValue(owner)) {
    if (!isLocalValue(result) || !isNumeric(result))
      return emitOpError("encoded Value field access produces a numeric local Value");
    if (logicalElement(result) != fieldType)
      return emitOpError("field result element must match the declared field type");
    return mlir::success();
  }
  auto resultSlice = mlir::dyn_cast<SliceType>(result);
  auto resultEncoding = resultSlice
                            ? mlir::dyn_cast<EncodingType>(resultSlice.getEncoding())
                            : EncodingType();
  if (!resultEncoding || resultEncoding.getKind() != "dense")
    return emitOpError("encoded View field access produces a dense numeric slice");
  if (resultEncoding.getFamily() != denseFamilyFor(fieldType))
    return emitOpError("field slice encoding must match the declared field type");
  return mlir::success();
}

mlir::LogicalResult ExtractOp::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return emitOpError("extract requires a shaped local Value and selectors");
  if (logicalElement(getInput().getType()) != logicalElement(getResult().getType()))
    return emitOpError("extract preserves its local element type");
  return verifyProjection(*this, getInput().getType(), getIndices(),
                          getSelectors(), getResult().getType(), true);
}

mlir::LogicalResult UpdateOp::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      getResult().getType() != getInput().getType())
    return emitOpError("update preserves one shaped state Value");
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return mlir::failure();
  if (logicalElement(getValue().getType()) !=
      mlir::cast<ValueType>(getInput().getType()).getElementType())
    return emitOpError("updated element type must match the state Value");
  return verifyProjection(*this, getInput().getType(), getIndices(),
                          getSelectors(), getValue().getType(), true);
}
