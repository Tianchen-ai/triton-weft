#ifndef WEFT_TARGET_X86_STORAGE_H
#define WEFT_TARGET_X86_STORAGE_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/EncodingStorage.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace weft::x86 {

// One numeric field element consists of byte fragments. This is a physical
// description of the declared storage relation, not a quantization formula.
struct Fragment {
  int64_t byte = 0, shift = 0, mask = 0, resultShift = 0, chunk = 1;
};
struct StorageLayout : kernel::EncodingStorage {
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
  // A runtime window may reuse one decode map when every aligned window has
  // exactly the same fragments, translated by a constant byte displacement.
  // Check both base-record and persistent-cohort addresses, including chunks.
  std::optional<std::pair<int64_t, int64_t>> translatedWindow(unsigned field,
                                                            int64_t extent) {
    auto dimensions = mlir::cast<mlir::DenseI64ArrayAttr>(base.getFieldShapes()[field]);
    if (dimensions.size() != 1 || extent <= 0 || dimensions[0] % extent ||
        dimensions[0] > 4096) return {};
    const int64_t count = dimensions[0];
    if (extent == count) return std::make_pair(int64_t{0}, int64_t{0});
    auto first = fragments(field, 0, false), next = fragments(field, extent, false);
    if (first.empty() || next.empty()) return {};
    const int64_t rawStride = next.front().byte - first.front().byte;
    const int64_t packedStride = address(next.front().byte).first - address(first.front().byte).first;
    for (int64_t start = extent; start < count; start += extent) {
      for (int64_t lane = 0; lane < extent; ++lane) {
        auto initial = fragments(field, lane, false);
        auto current = fragments(field, start + lane, false);
        if (initial.size() != current.size()) return {};
        for (auto [a, z] : llvm::zip(initial, current)) {
          auto aAddress = address(a.byte), zAddress = address(z.byte);
          if (a.shift != z.shift || a.mask != z.mask || a.resultShift != z.resultShift ||
              z.byte - a.byte != start / extent * rawStride ||
              zAddress.first - aAddress.first != start / extent * packedStride ||
              zAddress.second != aAddress.second) return {};
        }
      }
    }
    return std::make_pair(rawStride, packedStride);
  }
};

inline mlir::FailureOr<StorageLayout> storageLayout(mlir::ModuleOp module,
                                                   kernel::EncodingType encoding) {
  auto storage = kernel::encodingStorage(module, encoding);
  if (mlir::failed(storage)) return mlir::failure();
  return StorageLayout{std::move(*storage)};
}
} // namespace weft::x86
#endif
