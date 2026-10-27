#include "Verification.h"

using namespace weft::kernel;
using namespace weft::kernel::detail;

mlir::LogicalResult EncodingDeclOp::verify() {
  if (getKind() != "base")
    return emitOpError("encoding declaration must define a base layout");
  if (getLayoutIdentity() != getSymName())
    return emitOpError(
        "base Encoding layout identity must equal its declared family symbol");
  if (getBitOrder() != "lsb_first" && getBitOrder() != "msb_first")
    return emitOpError("bit_order must be lsb_first or msb_first");
  if (getByteOrder() != "little" && getByteOrder() != "big")
    return emitOpError("byte_order must be little or big");
  if (getAlignment() <= 0 || getElements() <= 0 || getStorageBits() <= 0)
    return emitOpError("alignment, elements, and storage_bits must be positive");
  if (failed(verifyStringArray(*this, getFieldNames(), "field_names")))
    return mlir::failure();
  const size_t count = getFieldNames().size();
  if (count == 0)
    return emitOpError("encoding declaration requires at least one field");
  if (getFieldTypes().size() != count || getFieldShapes().size() != count ||
      getFieldLayouts().size() != count ||
      getFieldBitOffsets().size() != count ||
      getFieldStorageBits().size() != count)
    return emitOpError("all encoding field arrays must have the same length");
  int64_t previousEnd = 0;
  int64_t joinedOffset = -1;
  int64_t joinedWidth = 0;
  int64_t joinedFields = 0;
  int64_t joinedNextRole = 0;
  int64_t joinedSize = 0;
  int64_t joinedLowBits = 0;
  std::string joinedOrder;
  llvm::SmallVector<std::pair<int64_t, int64_t>> spans;
  llvm::DenseSet<llvm::StringRef> fieldNames;
  for (size_t index = 0; index < count; ++index) {
    llvm::StringRef fieldName =
        mlir::cast<mlir::StringAttr>(getFieldNames()[index]).getValue();
    if (fieldName.empty() || !fieldNames.insert(fieldName).second)
      return emitOpError("encoding field names must be non-empty and unique");
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(getFieldTypes()[index]);
    auto shapeAttr =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(getFieldShapes()[index]);
    auto layouts = mlir::dyn_cast<mlir::ArrayAttr>(getFieldLayouts()[index]);
    if (!typeAttr || !shapeAttr || !layouts || layouts.empty() ||
        llvm::any_of(shapeAttr.asArrayRef(),
                     [](int64_t extent) { return extent <= 0; }))
      return emitOpError("field_types and field_shapes have invalid elements");
    if (!isCanonicalNumericElement(typeAttr.getValue()) ||
        typeAttr.getValue().isIndex())
      return emitOpError(
          "encoding fields require fixed-width signed, unsigned, or floating elements");
    auto kind = [](mlir::Attribute attribute) -> llvm::StringRef {
      auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
      auto value = dictionary ? dictionary.getAs<mlir::StringAttr>("kind")
                              : mlir::StringAttr();
      return value ? value.getValue() : llvm::StringRef();
    };
    bool sharedJoined = false;
    int64_t joinedRole = -1;
    if (kind(layouts[0]) == "natural") {
      if (layouts.size() != 1)
        return emitOpError("natural field layout cannot be combined");
      auto width = bitWidth(typeAttr.getValue());
      int64_t elements = 1;
      for (int64_t extent : shapeAttr.asArrayRef())
        elements *= extent;
      if (!width || getFieldStorageBits()[index] !=
                        static_cast<int64_t>(*width) * elements)
        return emitOpError(
            "natural field storage width must match its typed logical shape");
    } else if (kind(layouts[0]) == "joined") {
      if (layouts.size() != 1)
        return emitOpError("joined field layout cannot be combined");
      auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      auto group = joined.getAs<mlir::IntegerAttr>("size");
      auto fields = joined.getAs<mlir::IntegerAttr>("fields");
      auto lowBits = joined.getAs<mlir::IntegerAttr>("low_bits");
      auto role = joined.getAs<mlir::IntegerAttr>("role");
      auto order = joined.getAs<mlir::StringAttr>("order");
      auto integer = mlir::dyn_cast<mlir::IntegerType>(typeAttr.getValue());
      if (!group || !fields || !lowBits || !role || !order || !integer ||
          shapeAttr.size() != 1 || group.getInt() <= 0 || fields.getInt() < 2 ||
          role.getInt() < 0 || role.getInt() >= fields.getInt() ||
          shapeAttr[0] != 2 * group.getInt() || lowBits.getInt() <= 0 ||
          lowBits.getInt() >= integer.getWidth() ||
          lowBits.getInt() * fields.getInt() != 8 ||
          integer.getWidth() + integer.getWidth() - lowBits.getInt() != 8 ||
          (order.getValue() != "lo_first" && order.getValue() != "hi_first"))
        return emitOpError("invalid joined field layout");
      joinedRole = role.getInt();
      sharedJoined = true;
      const int64_t expectedStorage =
          group.getInt() * (fields.getInt() + 1) * 8;
      if (getFieldStorageBits()[index] != expectedStorage)
        return emitOpError(
            "joined field storage span must be uniquely implied by its layout");
    } else {
      if (layouts.size() != 2 || kind(layouts[0]) != "grouped" ||
          kind(layouts[1]) != "layered")
        return emitOpError(
            "field layout must be natural or grouped followed by layered");
      auto grouped = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      auto layered = mlir::cast<mlir::DictionaryAttr>(layouts[1]);
      auto group = grouped.getAs<mlir::IntegerAttr>("size");
      auto layer = layered.getAs<mlir::IntegerAttr>("size");
      auto order = layered.getAs<mlir::StringAttr>("order");
      if (!group || !layer || !order || group.getInt() <= 0 ||
          layer.getInt() <= 0 || group.getInt() % layer.getInt() ||
          (order.getValue() != "lo_first" &&
           order.getValue() != "hi_first"))
        return emitOpError("invalid grouped/layered field layout");
      auto integer = mlir::dyn_cast<mlir::IntegerType>(typeAttr.getValue());
      if (!integer || shapeAttr.size() != 1 ||
          shapeAttr[0] % group.getInt() ||
          integer.getWidth() * (group.getInt() / layer.getInt()) != 8)
        return emitOpError(
            "grouped/layered field must form one byte per layer position");
      if (getFieldStorageBits()[index] !=
          integer.getWidth() * shapeAttr[0])
        return emitOpError(
            "grouped/layered field storage span must match its complete logical field");
    }
    int64_t offset = getFieldBitOffsets()[index];
    int64_t width = getFieldStorageBits()[index];
    if (offset < 0 || width <= 0 || offset + width > getStorageBits())
      return emitOpError("encoding fields overlap or exceed storage_bits");
    if (sharedJoined) {
      auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      int64_t fields = joined.getAs<mlir::IntegerAttr>("fields").getInt();
      if (joinedRole == 0) {
        if (offset < previousEnd)
          return emitOpError("joined storage overlaps a previous field");
        joinedOffset = offset;
        joinedWidth = width;
        joinedFields = fields;
        joinedNextRole = 1;
        joinedSize = joined.getAs<mlir::IntegerAttr>("size").getInt();
        joinedLowBits = joined.getAs<mlir::IntegerAttr>("low_bits").getInt();
        joinedOrder = joined.getAs<mlir::StringAttr>("order").getValue().str();
        previousEnd = offset + width;
        spans.emplace_back(offset, offset + width);
      } else if (offset != joinedOffset || width != joinedWidth ||
                 fields != joinedFields || joinedRole != joinedNextRole++ ||
                 joined.getAs<mlir::IntegerAttr>("size").getInt() != joinedSize ||
                 joined.getAs<mlir::IntegerAttr>("low_bits").getInt() !=
                     joinedLowBits ||
                 joined.getAs<mlir::StringAttr>("order").getValue() !=
                     joinedOrder) {
        return emitOpError(
            "joined logical fields must be consecutive roles over one storage span");
      }
    } else {
      if (joinedOffset >= 0 && joinedNextRole != joinedFields)
        return emitOpError("joined storage is missing one or more logical fields");
      joinedOffset = -1;
      joinedNextRole = 0;
      if (offset < previousEnd)
        return emitOpError("encoding fields overlap");
      previousEnd = offset + width;
      spans.emplace_back(offset, offset + width);
    }
  }
  if (joinedOffset >= 0 && joinedNextRole != joinedFields)
    return emitOpError("joined storage is missing one or more logical fields");
  if (getPadding().size() % 3)
    return emitOpError("padding is a sequence of offset, width, fill triples");
  for (size_t index = 0; index < getPadding().size(); index += 3) {
    if (getPadding()[index] < 0 || getPadding()[index + 1] <= 0 ||
        getPadding()[index] + getPadding()[index + 1] > getStorageBits() ||
        getPadding()[index + 2] < 0 || getPadding()[index + 2] > 255)
      return emitOpError("padding triple is outside the declared layout");
    if (getPadding()[index] % 8 || getPadding()[index + 1] % 8)
      return emitOpError("padding byte ranges must be byte aligned");
    spans.emplace_back(getPadding()[index],
                       getPadding()[index] + getPadding()[index + 1]);
  }
  llvm::sort(spans);
  int64_t cursor = 0;
  for (auto [begin, end] : spans) {
    if (begin != cursor)
      return emitOpError("layout gaps must be represented by explicit padding");
    cursor = end;
  }
  if (cursor != getStorageBits())
    return emitOpError("fields and padding must cover storage_bits exactly");
  return mlir::success();
}

mlir::LogicalResult DeriveOp::verify() {
  if (getSourceFamily().empty() || getResultFamily().empty() ||
      getSourceFamily() == getResultFamily())
    return emitOpError("derive requires distinct source and result families");
  if (getLayoutIdentity().empty() ||
      failed(verifyStringArray(*this, getParameterNames(), "parameter_names")))
    return mlir::failure();
  if (getParameterNames().size() != getParameterValues().size() ||
      getParameterNames().empty())
    return emitOpError(
        "derive requires one typed value for every static parameter name");
  llvm::DenseSet<llvm::StringRef> parameterNames;
  for (auto [nameAttr, value] :
       llvm::zip(getParameterNames(), getParameterValues())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(nameAttr).getValue();
    if (name.empty() || !parameterNames.insert(name).second || value <= 0)
      return emitOpError(
          "derive parameter names are unique and values are positive");
  }
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != 1 || !mlir::isa<ViewType>(body.getArgument(0).getType()))
    return emitOpError("derive body takes one source View");
  auto source = mlir::cast<ViewType>(body.getArgument(0).getType()).getEncoding();
  auto sourceEncoding = mlir::cast<EncodingType>(source);
  if (sourceEncoding.getKind() != "base" ||
      sourceEncoding.getFamily() != getSourceFamily())
    return emitOpError("derive source View family does not match source_family");
  if (failed(verifyEncodingReference(*this, sourceEncoding)))
    return mlir::failure();
  auto yield = mlir::dyn_cast<DeriveYieldOp>(body.getTerminator());
  if (!yield)
    return emitOpError("derive body must end in derive_yield");
  auto resultEncoding = yield.getValue().getType().getEncoding();
  auto result = mlir::cast<EncodingType>(resultEncoding);
  if (result.getKind() != "derived_instance" ||
      result.getFamily() != getResultFamily() ||
      result.getLayoutIdentity() != getLayoutIdentity() ||
      result.getParameters().asArrayRef() != getParameterValues())
    return emitOpError(
        "derive result must be one concrete instance of its derived encoding family");
  if (failed(verifyEncodingReference(*this, result)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult DeriveYieldOp::verify() {
  if (!mlir::isa_and_nonnull<DeriveOp>(getOperation()->getParentOp()))
    return emitOpError("derive_yield terminates only a derive body");
  return mlir::success();
}

mlir::FailureOr<llvm::SmallVector<FieldInterleaveSpan>>
weft::kernel::fieldInterleaveSpans(
    int64_t storageBits, llvm::ArrayRef<int64_t> fieldBitOffsets,
    llvm::ArrayRef<int64_t> fieldStorageBits, llvm::ArrayRef<int64_t> padding,
    llvm::ArrayRef<int64_t> fieldChunks) {
  if (storageBits <= 0 || storageBits % 8 ||
      fieldChunks.size() != fieldStorageBits.size() ||
      fieldBitOffsets.size() != fieldStorageBits.size() ||
      llvm::any_of(fieldChunks, [](int64_t n) { return n <= 0; }))
    return mlir::failure();
  if (llvm::all_of(fieldChunks, [](int64_t n) { return n == 1; }))
    return llvm::SmallVector<FieldInterleaveSpan>{{0, storageBits / 8, 1}};
  llvm::SmallVector<FieldInterleaveSpan> spans;
  for (auto [offset, bits, chunk] :
       llvm::zip(fieldBitOffsets, fieldStorageBits, fieldChunks)) {
    if (offset < 0 || offset % 8 || bits <= 0 || bits % 8 ||
        bits / 8 % chunk || offset > storageBits - bits)
      return mlir::failure();
    spans.push_back({offset / 8, bits / 8, chunk});
  }
  if (padding.size() % 3)
    return mlir::failure();
  for (size_t i = 0; i < padding.size(); i += 3) {
    if (padding[i] < 0 || padding[i] % 8 || padding[i + 1] <= 0 ||
        padding[i + 1] % 8 || padding[i] > storageBits - padding[i + 1])
      return mlir::failure();
    spans.push_back({padding[i] / 8, padding[i + 1] / 8, 1});
  }
  llvm::sort(spans, [](const auto &a, const auto &b) {
    return std::tie(a.byteOffset, a.byteCount, a.chunkBytes) <
           std::tie(b.byteOffset, b.byteCount, b.chunkBytes);
  });
  llvm::SmallVector<FieldInterleaveSpan> result;
  int64_t end = 0;
  for (const auto &span : spans) {
    if (!result.empty() && span.byteOffset == result.back().byteOffset &&
        span.byteCount == result.back().byteCount &&
        span.chunkBytes == result.back().chunkBytes)
      continue;
    if (span.byteOffset != end)
      return mlir::failure();
    result.push_back(span);
    end += span.byteCount;
  }
  if (end != storageBits / 8)
    return mlir::failure();
  return result;
}

mlir::LogicalResult InterleaveOp::verify() {
  if (getRows() <= 0 || !sameDomain(getView().getType(), getResult().getType()))
    return emitOpError("interleave requires positive rows and preserves the View domain");
  if (mlir::cast<EncodingType>(getResult().getType().getEncoding()).getKind() !=
      "derived_instance")
    return emitOpError("interleave in derive produces a concrete derived instance");
  auto derive = mlir::dyn_cast_or_null<DeriveOp>(getOperation()->getParentOp());
  if (!derive)
    return emitOpError("interleave is valid only inside a derive body");
  auto resultEncoding =
      mlir::cast<EncodingType>(getResult().getType().getEncoding());
  if (derive.getParameterValues().size() != 1 ||
      derive.getParameterValues()[0] != getRows() ||
      resultEncoding.getParameters().asArrayRef() !=
          derive.getParameterValues() ||
      resultEncoding.getLayoutIdentity() != derive.getLayoutIdentity())
    return emitOpError(
        "interleave rows and result identity must match its derive instance");
  auto source = mlir::SymbolTable::lookupNearestSymbolFrom<EncodingDeclOp>(
      *this, mlir::StringAttr::get(getContext(), derive.getSourceFamily()));
  if (!source || mlir::failed(fieldInterleaveSpans(
                     source.getStorageBits(), source.getFieldBitOffsets(),
                     source.getFieldStorageBits(), source.getPadding(),
                     getFieldChunks())))
    return emitOpError(
        "field interleave chunks must partition the complete byte-addressable source storage without conflicting aliases");
  return mlir::success();
}
