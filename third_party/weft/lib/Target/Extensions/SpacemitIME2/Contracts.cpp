#include "Weft/Target/RISCVFragment.h"
#include "Weft/Target/RISCVExtensions.h"

namespace {
llvm::ArrayRef<weft::RISCVFragmentCapability> fragmentContracts() {
  using namespace weft;
  // A100 base integer MMA: A[M,16] B[N,16]^T, with an i32 pair
  // of RVV registers for the full 8x8 result. Partial M exposes only
  // initialized rows; it does not change the instruction or its resources.
  static const std::vector<RISCVFragmentCapability> capabilities = [] {
    std::vector<RISCVFragmentCapability> result;
    for (unsigned rows : {8U, 4U, 1U}) {
      result.push_back({
          "spacemit-ime2-i8-m" + std::to_string(rows) + "-mma",
          "spacemit-ime2", 64, 1024,
          RISCVFragmentSignedness::Signed, RISCVFragmentSignedness::Signed,
          8, 8, 32, 8, 8, 16, 1, 1, 2,
          {"ime2-i8-m" + std::to_string(rows) + "-k16", {0, 1}, rows, 16,
           "row_major", "row_major", 8, 16},
          {"ime2-i8-n8-k16-transposed", {1, 0}, 8, 16,
           "row_major", "row_major", 8, 16},
          {"ime2-i32-m" + std::to_string(rows) + "-n8", {0, 1}, rows, 8,
           "row_major", "row_major", 32, 32},
          1, 1, {"t0", "vl", "vtype"}, true, false,
          0, 0, 0, 1, 2, RISCVFragmentSignedness::Signed,
          rows, rows == 8 ? 4U : 1U, 4, 4, "spacemit-ime2-vupack128", "spacemit-ime2-vpack-rows"});
    }
    return result;
  }();
  return capabilities;
}
} // namespace

namespace weft {
const RISCVExtension &getSpacemitIME2Extension() {
  // Linux on K3 reports the A100 architectural identity but no vendor
  // extension token in /proc/cpuinfo. The exact identity and measured VLEN
  // establish this fixed ISA profile; X100 cores have a different marchid.
  static const RISCVExtension extension{
      "spacemit-ime2", "matrix",
      "A100 signed i8 matrix atoms with i32 accumulation in shared RVV registers",
      {{64, 1024, 0x710, 0x8000000041000002, ""}}, fragmentContracts()};
  return extension;
}
} // namespace weft
