#ifndef WEFT_TARGET_X86_STORAGE_H
#define WEFT_TARGET_X86_STORAGE_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace weft::x86 {

// One numeric field element consists of byte fragments. This is a physical
// description of the declared storage relation, not a quantization formula.
struct Fragment {
  int64_t byte = 0, shift = 0, mask = 0, resultShift = 0, chunk = 1;
};
struct StorageLayout {
  kernel::EncodingDeclOp base;
  int64_t rows = 1;
  llvm::SmallVector<kernel::FieldInterleaveSpan> spans;

  int64_t bytes() { return base.getStorageBits() / 8; }
  int64_t elements() { return base.getElements(); }
  std::pair<int64_t, int64_t> address(int64_t byte) const {
    if (rows == 1) return {byte, 1};
    for (const auto &span : spans) {
      if (byte < span.byteOffset || byte >= span.byteOffset + span.byteCount) continue;
      int64_t offset = byte - span.byteOffset;
      return {span.byteOffset * rows + offset / span.chunkBytes * span.chunkBytes * rows +
                  offset % span.chunkBytes, span.chunkBytes};
    }
    llvm_unreachable("verified interleave spans cover every record byte");
  }
  llvm::SmallVector<Fragment> fragments(unsigned field, int64_t index,
                                        bool interleaved = true) {
    unsigned width = mlir::cast<mlir::TypeAttr>(base.getFieldTypes()[field]).getValue().getIntOrFloatBitWidth();
    int64_t begin = base.getFieldBitOffsets()[field];
    auto layouts = mlir::cast<mlir::ArrayAttr>(base.getFieldLayouts()[field]);
    auto layout = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
    auto number = [](mlir::DictionaryAttr attr, llvm::StringRef name) {
      return attr.getAs<mlir::IntegerAttr>(name).getInt();
    };
    llvm::SmallVector<Fragment> result;
    auto append = [&](int64_t bit, int64_t count, int64_t destination) {
      while (count) {
        int64_t take = std::min<int64_t>(count, 8 - bit % 8);
        int64_t shift = base.getBitOrder() == "lsb_first" ? bit % 8 : 8 - bit % 8 - take;
        auto [byte, chunk] = interleaved ? address(bit / 8) : std::make_pair(bit / 8, int64_t{1});
        result.push_back({byte, shift, (int64_t{1} << take) - 1, destination, chunk});
        bit += take; destination += take; count -= take;
      }
    };
    auto kind = layout.getAs<mlir::StringAttr>("kind").getValue();
    if (kind == "natural") {
      if (width % 8 == 0 && begin % 8 == 0 && base.getByteOrder() == "big") {
        for (unsigned byte = 0; byte < width / 8; ++byte)
          append(begin + index * width + (width / 8 - byte - 1) * 8, 8, byte * 8);
      } else append(begin + index * width, width, 0);
    } else if (kind == "grouped") {
      auto layer = mlir::cast<mlir::DictionaryAttr>(layouts[1]);
      int64_t group = number(layout, "size"), lanes = number(layer, "size");
      int64_t position = index % group / lanes;
      if (layer.getAs<mlir::StringAttr>("order").getValue() == "hi_first") position = group / lanes - position - 1;
      append(begin + (index / group * lanes + index % lanes) * 8 + position * width, width, 0);
    } else {
      assert(kind == "joined" && "canonical verifier closes the field vocabulary");
      int64_t group = number(layout, "size"), fields = number(layout, "fields");
      int64_t low = number(layout, "low_bits"), role = number(layout, "role");
      int64_t physicalRole = layout.getAs<mlir::StringAttr>("order").getValue() == "lo_first" ? role : fields - role - 1;
      if (index < group) append(begin + (role * group + index) * 8, width, 0);
      else {
        append(begin + (fields * group + index - group) * 8 + physicalRole * low, low, 0);
        append(begin + (role * group + index - group) * 8 + width, width - low, low);
      }
    }
    return result;
  }
  std::optional<int64_t> affineFieldOffset(unsigned field) {
    unsigned width = mlir::cast<mlir::TypeAttr>(base.getFieldTypes()[field]).getValue().getIntOrFloatBitWidth();
    if (rows != 1 || width % 8) return {};
    int64_t count = 1;
    for (int64_t extent : mlir::cast<mlir::DenseI64ArrayAttr>(base.getFieldShapes()[field]).asArrayRef()) count *= extent;
    int64_t elementBytes = width / 8;
    int64_t begin = base.getFieldBitOffsets()[field] / 8;
    if (bytes() % elementBytes || base.getFieldBitOffsets()[field] % width) return {};
    for (int64_t index = 0; index < count; ++index) {
      auto pieces = fragments(field, index);
      if (pieces.size() != static_cast<size_t>(elementBytes)) return {};
      for (auto [byte, piece] : llvm::enumerate(pieces))
        if (piece.byte != begin + index * elementBytes + static_cast<int64_t>(byte) ||
            piece.shift != 0 || piece.mask != 255 || piece.resultShift != static_cast<int64_t>(byte) * 8)
          return {};
    }
    return begin / elementBytes;
  }
  bool isDenseField(unsigned field) {
    auto offset = affineFieldOffset(field);
    if (!offset || *offset) return false;
    unsigned width = mlir::cast<mlir::TypeAttr>(base.getFieldTypes()[field]).getValue().getIntOrFloatBitWidth();
    int64_t count = 1;
    for (int64_t extent : mlir::cast<mlir::DenseI64ArrayAttr>(base.getFieldShapes()[field]).asArrayRef()) count *= extent;
    return count * (width / 8) == bytes();
  }
};

inline mlir::FailureOr<StorageLayout> storageLayout(mlir::ModuleOp module,
                                                   kernel::EncodingType encoding) {
  StorageLayout result;
  llvm::StringRef family = encoding.getFamily();
  kernel::InterleaveOp interleave;
  if (encoding.getKind() == "derived_instance") {
    for (auto derive : module.getOps<kernel::DeriveOp>()) {
      if (derive.getResultFamily() != family || derive.getLayoutIdentity() != encoding.getLayoutIdentity() ||
          derive.getParameterValues() != encoding.getParameters().asArrayRef()) continue;
      auto &body = derive.getBody().front();
      if (body.getOperations().size() != 2 || !(interleave = mlir::dyn_cast<kernel::InterleaveOp>(body.front())))
        return derive.emitError("x86 artifact layout requires the declared interleave program");
      family = derive.getSourceFamily();
      result.rows = interleave.getRows();
      break;
    }
    if (!interleave) return mlir::failure();
  }
  for (auto declaration : module.getOps<kernel::EncodingDeclOp>())
    if (declaration.getSymName() == family) result.base = declaration;
  if (!result.base || result.base.getStorageBits() % 8) return mlir::failure();
  if (interleave) {
    auto spans = kernel::fieldInterleaveSpans(result.base.getStorageBits(), result.base.getFieldBitOffsets(),
        result.base.getFieldStorageBits(), result.base.getPadding(), interleave.getFieldChunks());
    if (mlir::failed(spans)) return mlir::failure();
    result.spans = std::move(*spans);
  }
  return result;
}
} // namespace weft::x86
#endif
