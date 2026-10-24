#include "Weft/Analysis/LogicalAxes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace weft::analysis;

std::optional<llvm::SmallVector<unsigned>> weft::analysis::axisPermutation(
    llvm::ArrayRef<int64_t> axes, llvm::ArrayRef<int64_t> order) {
  if (axes.size() != order.size()) return {};
  llvm::SmallDenseSet<int64_t> unique;
  for (int64_t axis : axes) if (!unique.insert(axis).second) return {};
  unique.clear();
  llvm::SmallVector<unsigned> result;
  for (int64_t axis : order) {
    auto found = llvm::find(axes, axis);
    if (found == axes.end() || !unique.insert(axis).second) return {};
    result.push_back(found - axes.begin());
  }
  return result;
}

std::optional<MatrixAxes> weft::analysis::matrixAxes(
    LogicalDomain left, LogicalDomain right, LogicalDomain output,
    llvm::ArrayRef<int64_t> reductionAxes) {
  if (reductionAxes.size() != 1 || output.axes.size() != 2 ||
      output.shape.size() != 2 || output.axes[0] == output.axes[1] ||
      llvm::is_contained(output.axes, reductionAxes[0])) return {};
  auto position = [&](LogicalDomain input, unsigned side) -> std::optional<unsigned> {
    if (input.axes.size() != 2 || input.shape.size() != 2 ||
        llvm::any_of(input.shape, [](int64_t extent) { return extent <= 0; })) return {};
    auto found = llvm::find(input.axes, reductionAxes[0]);
    if (found == input.axes.end()) return {};
    unsigned reduced = found - input.axes.begin();
    if (input.axes[1 - reduced] != output.axes[side] ||
        input.shape[1 - reduced] != output.shape[side]) return {};
    return reduced;
  };
  auto lhs = position(left, 0), rhs = position(right, 1);
  if (!lhs || !rhs || left.shape[*lhs] != right.shape[*rhs]) return {};
  return MatrixAxes{*lhs, *rhs, output.shape[0], output.shape[1], left.shape[*lhs]};
}

std::optional<GroupedAxes> weft::analysis::groupedAxes(
    llvm::ArrayRef<LogicalDomain> inputs, LogicalDomain output, int64_t group) {
  auto valid = [](LogicalDomain domain) {
    if (domain.shape.empty() || domain.shape.size() != domain.axes.size()) return false;
    if (llvm::any_of(domain.shape, [](int64_t extent) { return extent <= 0; })) return false;
    llvm::SmallDenseSet<int64_t> ids;
    for (int64_t axis : domain.axes)
      if (!ids.insert(axis).second) return false;
    return true;
  };
  if (inputs.empty() || group <= 0 || !valid(output) || output.shape.back() <= 0) return {};
  GroupedAxes result{output.axes.back(), group, output.shape.back(), {}};
  llvm::SmallDenseSet<int64_t> used;
  for (LogicalDomain input : inputs) {
    if (!valid(input) || input.axes.back() != result.axis || input.shape.back() <= 0 ||
        input.shape.back() % group || input.shape.back() / group != result.groups) return {};
    llvm::SmallVector<unsigned> mapping;
    for (auto [position, axis] : llvm::enumerate(input.axes)) {
      auto outputPosition = llvm::find(output.axes, axis);
      if (outputPosition == output.axes.end()) return {};
      unsigned index = outputPosition - output.axes.begin();
      if (axis != result.axis && input.shape[position] != 1 && input.shape[position] != output.shape[index]) return {};
      mapping.push_back(index);
      used.insert(axis);
    }
    result.inputToOutput.push_back(std::move(mapping));
  }
  if (used.size() != output.axes.size()) return {};
  return result;
}
