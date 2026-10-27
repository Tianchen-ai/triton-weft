#pragma once

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/Atomic.h"
#include "Weft/Dialect/Kernel/IR/ReductionRegion.h"
#include "Weft/Dialect/Kernel/IR/StrideABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Weft/Dialect/Kernel/IR/SubviewBounds.h"
#include "Weft/Dialect/Kernel/IR/ShapeBindings.h"
#include <functional>
#include <limits>
#include <optional>
#include <tuple>

#include "Weft/Dialect/Kernel/IR/TypeQueries.h"

namespace weft::kernel::detail {
bool isScalar(mlir::Type type);
bool isEncoding(mlir::Type type);
bool isViewLike(mlir::Type type);
bool isLocalValue(mlir::Type type);
bool isCanonicalNumericElement(mlir::Type type);
bool isCanonicalProgramType(mlir::Type type);
bool isNumeric(mlir::Type type);
std::optional<unsigned> bitWidth(mlir::Type type);
std::string denseFamilyFor(mlir::Type type);
EncodingDeclOp findEncodingDeclaration(mlir::Operation *operation,
                                       llvm::StringRef family);
mlir::LogicalResult verifyEncodingReference(mlir::Operation *operation,
                                            EncodingType encoding);
mlir::LogicalResult verifyEncodingInType(mlir::Operation *operation,
                                         mlir::Type type);
llvm::StringRef baseFamilyFor(mlir::Operation *operation,
                              EncodingType encoding);
mlir::LogicalResult verifyShape(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> axes,
    bool allowEmpty);
mlir::LogicalResult verifyStringArray(mlir::Operation *operation,
                                      mlir::ArrayAttr values,
                                      llvm::StringRef name);
mlir::LogicalResult verifyPointwise(mlir::Operation *operation,
                                    mlir::Type lhs, mlir::Type rhs,
                                    mlir::Type result);
mlir::LogicalResult verifyProjection(mlir::Operation *operation,
                                     mlir::Type base,
                                     mlir::ValueRange indices,
                                     mlir::ArrayAttr selectors,
                                     mlir::Type projected,
                                     bool allowGather);
mlir::Type promotedElement(mlir::Type lhs, mlir::Type rhs);
mlir::LogicalResult verifyContractLike(mlir::Operation *operation,
                                       mlir::Value lhs, mlir::Value rhs,
                                       llvm::ArrayRef<int64_t> over,
                                       std::optional<mlir::Type> accType,
                                       mlir::Type result);
} // namespace weft::kernel::detail
