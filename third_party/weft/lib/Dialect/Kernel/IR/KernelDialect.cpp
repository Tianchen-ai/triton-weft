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

using namespace weft::kernel;

#include "Weft/Dialect/Kernel/IR/KernelOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelOps.cpp.inc"

void WEFTKernelDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Weft/Dialect/Kernel/IR/KernelTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Kernel/IR/KernelOps.cpp.inc"
      >();
}
