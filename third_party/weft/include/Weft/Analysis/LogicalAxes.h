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

struct MatrixAxes {
  unsigned leftReduction, rightReduction;
  int64_t rows, columns, terms;
};

// A closed rank-two contraction, independent of either operand's storage order.
// Dynamic domains remain the responsibility of their runtime coordinate plan.
std::optional<MatrixAxes> matrixAxes(LogicalDomain left, LogicalDomain right,
                                     LogicalDomain output,
                                     llvm::ArrayRef<int64_t> reductionAxes);
} // namespace weft::analysis
#endif
