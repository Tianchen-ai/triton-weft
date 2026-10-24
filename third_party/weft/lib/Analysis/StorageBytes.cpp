#include "Weft/Analysis/StorageBytes.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
using namespace weft::analysis;
namespace kernel = weft::kernel;

namespace {
std::optional<int64_t> product(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b)) return {};
  return a * b;
}
std::optional<int64_t> sum(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) return {};
  return a + b;
}
FieldBitLayout fieldLayout(kernel::EncodingDeclOp base, unsigned field) {
  return fieldBitLayout(cast<TypeAttr>(base.getFieldTypes()[field]).getValue(),
                        cast<ArrayAttr>(base.getFieldLayouts()[field]));
}
int64_t fieldElements(kernel::EncodingDeclOp base, unsigned field) {
  int64_t count = 1;
  for (int64_t extent : cast<DenseI64ArrayAttr>(base.getFieldShapes()[field]).asArrayRef()) count *= extent;
  return count;
}
}

FieldBitLayout weft::analysis::fieldBitLayout(Type element, ArrayAttr layouts) {
  FieldBitLayout result;
  result.elementBits = element.getIntOrFloatBitWidth();
  auto layout = cast<DictionaryAttr>(layouts[0]);
  auto number = [](DictionaryAttr attr, StringRef name) { return attr.getAs<IntegerAttr>(name).getInt(); };
  auto kind = layout.getAs<StringAttr>("kind").getValue();
  if (kind == "natural") result.mapping = "natural";
  else if (kind == "grouped") {
    auto layered = cast<DictionaryAttr>(layouts[1]);
    result.mapping = "grouped_layered";
    result.group = number(layout, "size"); result.layer = number(layered, "size");
    result.highFirst = layered.getAs<StringAttr>("order").getValue() == "hi_first";
  } else {
    assert(kind == "joined" && "verified Encoding field vocabulary");
    result.mapping = "joined"; result.group = number(layout, "size");
    result.joinedFields = number(layout, "fields"); result.joinedLowBits = number(layout, "low_bits");
    result.joinedRole = number(layout, "role");
    result.highFirst = layout.getAs<StringAttr>("order").getValue() == "hi_first";
  }
  return result;
}

std::optional<int64_t> FieldBitLayout::contiguousBit(int64_t index) const {
  if (elementBits <= 0 || index < 0) return {};
  if (mapping == "natural") return product(index, elementBits);
  if (mapping != "grouped_layered" || group <= 0 || layer <= 0 || group % layer) return {};
  int64_t layers = group / layer, within = index % group;
  int64_t selected = highFirst ? layers - 1 - within / layer : within / layer;
  auto groupBits = product(group, elementBits), stride = product(layers, elementBits);
  if (!groupBits || !stride) return {};
  auto prefix = product(index / group, *groupBits), position = product(within % layer, *stride);
  auto bits = product(selected, elementBits);
  if (!prefix || !position || !bits) return {};
  auto offset = sum(*prefix, *position);
  return offset ? sum(*offset, *bits) : std::nullopt;
}

std::optional<llvm::SmallVector<FieldBitLayout::Slice>> FieldBitLayout::slices(int64_t index) const {
  if (mapping != "joined") {
    auto bit = contiguousBit(index);
    if (!bit || !sum(*bit, elementBits)) return {};
    return llvm::SmallVector<Slice>{{*bit, elementBits, 0}};
  }
  if (index < 0 || group <= 0 || index / group >= 2 || joinedFields <= 0 ||
      joinedRole < 0 || joinedRole >= joinedFields || joinedLowBits <= 0 || joinedLowBits >= elementBits ||
      elementBits > 8 || !product(joinedFields, joinedLowBits) || joinedFields * joinedLowBits != 8 ||
      elementBits + (elementBits - joinedLowBits) != 8) return {};
  auto roleBase = product(joinedRole, group);
  if (!roleBase) return {};
  auto sourceByte = sum(*roleBase, index < group ? index : index - group);
  auto sourceBit = sourceByte ? product(*sourceByte, 8) : std::nullopt;
  if (!sourceBit) return {};
  if (index < group) {
    if (!sum(*sourceBit, elementBits)) return {};
    return llvm::SmallVector<Slice>{{*sourceBit, elementBits, 0}};
  }
  auto tailBase = product(joinedFields, group);
  auto tailByte = tailBase ? sum(*tailBase, index - group) : std::nullopt;
  auto tailBit = tailByte ? product(*tailByte, 8) : std::nullopt;
  int64_t role = highFirst ? joinedFields - joinedRole - 1 : joinedRole;
  auto low = tailBit ? sum(*tailBit, role * joinedLowBits) : std::nullopt;
  auto high = sum(*sourceBit, elementBits);
  if (!low || !high || !sum(*low, joinedLowBits) || !sum(*high, elementBits - joinedLowBits)) return {};
  return llvm::SmallVector<Slice>{{*low, joinedLowBits, 0}, {*high, elementBits - joinedLowBits, joinedLowBits}};
}

std::pair<int64_t, int64_t> EncodingBytes::address(int64_t byte) const {
  if (rows == 1) return {byte, 1};
  for (const auto &span : spans) {
    if (byte < span.byteOffset || byte >= span.byteOffset + span.byteCount) continue;
    int64_t offset = byte - span.byteOffset;
    return {span.byteOffset * rows + offset / span.chunkBytes * span.chunkBytes * rows + offset % span.chunkBytes,
            span.chunkBytes};
  }
  llvm_unreachable("verified interleave spans cover every record byte");
}

std::optional<llvm::SmallVector<StorageByteFragment>> weft::analysis::fieldByteFragments(
    const FieldBitLayout &layout, int64_t begin, int64_t index, StringRef bitOrder, StringRef byteOrder) {
  if (begin < 0 || (bitOrder != "lsb_first" && bitOrder != "msb_first") ||
      (byteOrder != "little" && byteOrder != "big")) return {};
  auto pieces = layout.slices(index);
  if (!pieces) return {};
  if (layout.mapping == "natural" && layout.elementBits % 8 == 0 && begin % 8 == 0 && byteOrder == "big") {
    pieces->clear();
    for (int64_t byte = 0; byte < layout.elementBits / 8; ++byte)
      pieces->push_back({index * layout.elementBits + (layout.elementBits / 8 - byte - 1) * 8, 8, byte * 8});
  }
  llvm::SmallVector<StorageByteFragment> result;
  for (auto slice : *pieces) {
    auto end = sum(slice.bit, slice.count);
    if (!end || !sum(begin, *end)) return {};
    int64_t bit = begin + slice.bit, count = slice.count, destination = slice.resultShift;
    while (count) {
      int64_t take = std::min<int64_t>(count, 8 - bit % 8);
      int64_t shift = bitOrder == "lsb_first" ? bit % 8 : 8 - bit % 8 - take;
      result.push_back({bit / 8, shift, (int64_t{1} << take) - 1, destination, 1});
      bit += take; destination += take; count -= take;
    }
  }
  return result;
}

llvm::SmallVector<StorageByteFragment> EncodingBytes::fragments(unsigned field, int64_t index, bool interleaved) {
  auto result = fieldByteFragments(fieldLayout(base, field), base.getFieldBitOffsets()[field], index,
                                   base.getBitOrder(), base.getByteOrder());
  assert(result && "verified Encoding field coordinate must have a complete byte relation");
  if (interleaved)
    for (auto &fragment : *result) {
      auto [byte, chunk] = address(fragment.byte);
      fragment.byte = byte; fragment.chunk = chunk;
    }
  return *result;
}

std::optional<int64_t> EncodingBytes::affineFieldOffset(unsigned field) {
  auto layout = fieldLayout(base, field);
  int64_t width = layout.elementBits;
  if (rows != 1 || width % 8) return {};
  int64_t elementBytes = width / 8, begin = base.getFieldBitOffsets()[field] / 8;
  if (base.getStorageBits() / 8 % elementBytes || base.getFieldBitOffsets()[field] % width) return {};
  // Prove the complete coordinate family, without enumerating a potentially
  // large field or bounding correctness by a handful of sample coordinates.
  bool natural = layout.mapping == "natural" ||
      (layout.mapping == "grouped_layered" && layout.group == layout.layer && width == 8);
  if (!natural || (width > 8 && base.getByteOrder() != "little")) return {};
  return begin / elementBytes;
}
bool EncodingBytes::isDenseField(unsigned field) {
  auto offset = affineFieldOffset(field);
  return offset && !*offset && fieldElements(base, field) * (fieldLayout(base, field).elementBits / 8) == base.getStorageBits() / 8;
}
std::optional<std::pair<int64_t, int64_t>> EncodingBytes::translatedWindow(unsigned field, int64_t extent) {
  auto shape = cast<DenseI64ArrayAttr>(base.getFieldShapes()[field]);
  if (shape.size() != 1 || extent <= 0 || shape[0] % extent || shape[0] > 4096) return {};
  if (extent == shape[0]) return std::make_pair(int64_t{0}, int64_t{0});
  auto first = fragments(field, 0, false), next = fragments(field, extent, false);
  if (first.empty() || next.empty()) return {};
  int64_t rawStride = next.front().byte - first.front().byte;
  int64_t packedStride = address(next.front().byte).first - address(first.front().byte).first;
  for (int64_t start = extent; start < shape[0]; start += extent)
    for (int64_t lane = 0; lane < extent; ++lane) {
      auto initial = fragments(field, lane, false), current = fragments(field, start + lane, false);
      if (initial.size() != current.size()) return {};
      for (auto [a, z] : llvm::zip(initial, current)) {
        auto aa = address(a.byte), za = address(z.byte);
        if (a.shift != z.shift || a.mask != z.mask || a.resultShift != z.resultShift ||
            z.byte - a.byte != start / extent * rawStride || za.first - aa.first != start / extent * packedStride ||
            za.second != aa.second) return {};
      }
    }
  return std::make_pair(rawStride, packedStride);
}

FailureOr<EncodingBytes> weft::analysis::encodingBytes(ModuleOp module, kernel::EncodingType encoding) {
  auto storage = kernel::encodingStorage(module, encoding);
  if (failed(storage)) return failure();
  return EncodingBytes{std::move(*storage)};
}
