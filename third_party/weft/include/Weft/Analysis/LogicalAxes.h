#ifndef WEFT_ANALYSIS_LOGICAL_AXES_H
#define WEFT_ANALYSIS_LOGICAL_AXES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace weft::analysis {
struct LogicalDomain {
  llvm::ArrayRef<int64_t> shape;
  llvm::ArrayRef<int64_t> axes;
};

struct GroupedAxes {
  int64_t axis;
  int64_t group;
  int64_t groups;
  // For each input dimension, the matching output dimension. A unit input
  // extent broadcasts, while the grouped dimension maps g -> g*group+member.
  llvm::SmallVector<llvm::SmallVector<unsigned>> inputToOutput;
};

std::optional<GroupedAxes> groupedAxes(llvm::ArrayRef<LogicalDomain> inputs,
                                      LogicalDomain output, int64_t group);
} // namespace weft::analysis
#endif
