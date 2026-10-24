#ifndef WEFT_ANALYSIS_REDUCTIONIDENTITY_H
#define WEFT_ANALYSIS_REDUCTIONIDENTITY_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <limits>

namespace weft::analysis {

// Identity for the declared closed operation, including all-NaN extrema and
// finite-width integer order. Signless machine types supply their source sign.
inline mlir::TypedAttr reductionIdentity(mlir::Type element, llvm::StringRef kind,
                                         bool unsignedInteger = false) {
  if (kind != "add" && kind != "max" && kind != "min") return {};
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element)) {
    unsigned bits = integer.getWidth();
    bool uns = integer.isUnsigned() || (integer.isSignless() && unsignedInteger);
    llvm::APInt value(bits, 0);
    if (kind == "max") value = uns ? llvm::APInt::getMinValue(bits) : llvm::APInt::getSignedMinValue(bits);
    if (kind == "min") value = uns ? llvm::APInt::getMaxValue(bits) : llvm::APInt::getSignedMaxValue(bits);
    return mlir::IntegerAttr::get(element, value);
  }
  if (mlir::isa<mlir::FloatType>(element))
    return mlir::FloatAttr::get(element, kind == "add" ? 0.0 : std::numeric_limits<double>::quiet_NaN());
  return {};
}

} // namespace weft::analysis
#endif
