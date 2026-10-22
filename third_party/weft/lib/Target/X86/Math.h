#ifndef WEFT_X86_MATH_H
#define WEFT_X86_MATH_H

#include "Weft/Target/X86Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace weft {
mlir::Value widenX86ScalarHalf(mlir::ModuleOp module, mlir::OpBuilder &builder,
                              mlir::Location loc, mlir::Value input,
                              const X86Target &target);
}
#endif
