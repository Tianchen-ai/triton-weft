#include "Weft/Dialect/Kernel/IR/TypeQueries.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include <limits>

namespace weft::kernel {
mlir::Type logicalElement(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getElementType();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getEncoding();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getEncoding();
  return type;
}

llvm::ArrayRef<int64_t> logicalShape(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getShape().asArrayRef();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getShape().asArrayRef();
  return {};
}

llvm::ArrayRef<int64_t> logicalAxes(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getAxisIds().asArrayRef();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getAxisIds().asArrayRef();
  return {};
}

bool sameDomain(mlir::Type lhs, mlir::Type rhs) {
  return logicalShape(lhs) == logicalShape(rhs) &&
         logicalAxes(lhs) == logicalAxes(rhs);
}

std::optional<int64_t> staticElementCount(mlir::Type type) {
  int64_t count = 1;
  for (int64_t extent : logicalShape(type)) {
    if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent)
      return std::nullopt;
    count *= extent;
  }
  return count;
}

bool isUnitAxisProjection(
    llvm::ArrayRef<int64_t> inputShape, llvm::ArrayRef<int64_t> inputAxes,
    llvm::ArrayRef<int64_t> resultShape, llvm::ArrayRef<int64_t> resultAxes,
    llvm::ArrayRef<int64_t> inputOrder) {
  if (inputShape.size() != inputAxes.size() ||
      resultShape.size() != resultAxes.size() || inputOrder != inputAxes)
    return false;
  size_t source = 0;
  for (auto [extent, axis] : llvm::zip(resultShape, resultAxes)) {
    if (extent == 1) continue;
    while (source < inputShape.size() && inputShape[source] == 1) ++source;
    if (source == inputShape.size() || inputShape[source] != extent ||
        inputAxes[source] != axis)
      return false;
    ++source;
  }
  while (source < inputShape.size() && inputShape[source] == 1) ++source;
  return source == inputShape.size();
}

} // namespace weft::kernel
