#ifndef WEFT_ANALYSIS_STORAGE_BYTES_H
#define WEFT_ANALYSIS_STORAGE_BYTES_H

#include "Weft/Dialect/Kernel/IR/EncodingStorage.h"
#include <optional>

namespace weft::analysis {

// The bit relation belongs to the declared Encoding. It has no lane, register,
// machine integer width, quantization formula, or target instruction identity.
struct FieldBitLayout {
  llvm::StringRef mapping;
  int64_t elementBits = 0;
  int64_t group = 0, layer = 0;
  int64_t joinedFields = 0, joinedLowBits = 0, joinedRole = 0;
  bool highFirst = false;

  // A contiguous logical bit span in one base record, before byte/bit order and
  // persistent interleave. Joined elements can have two separate spans.
  struct Slice { int64_t bit = 0, count = 0, resultShift = 0; };
  std::optional<llvm::SmallVector<Slice>> slices(int64_t logicalIndex) const;
  std::optional<int64_t> contiguousBit(int64_t logicalIndex) const;
};

// Both Canonical and target declarations retain these verified field attrs.
FieldBitLayout fieldBitLayout(mlir::Type element, mlir::ArrayAttr layouts);

// One numerical field element may be supplied by several record bytes. A
// fragment explicitly names the source bit range and destination bit range.
struct StorageByteFragment {
  int64_t byte = 0, shift = 0, mask = 0, resultShift = 0, chunk = 1;
};

std::optional<llvm::SmallVector<StorageByteFragment>> fieldByteFragments(
    const FieldBitLayout &layout, int64_t bitOffset, int64_t logicalIndex,
    llvm::StringRef bitOrder, llvm::StringRef byteOrder);

struct EncodingBytes : kernel::EncodingStorage {
  std::pair<int64_t, int64_t> address(int64_t byte) const;
  llvm::SmallVector<StorageByteFragment> fragments(unsigned field, int64_t index,
                                                   bool interleaved = true);
  std::optional<int64_t> affineFieldOffset(unsigned field);
  bool isDenseField(unsigned field);
  std::optional<std::pair<int64_t, int64_t>> translatedWindow(unsigned field, int64_t extent);
};

mlir::FailureOr<EncodingBytes> encodingBytes(mlir::ModuleOp module, kernel::EncodingType encoding);

} // namespace weft::analysis
#endif
