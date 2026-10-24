#include "Weft/Analysis/LogicalAxes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace weft::analysis;

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
