#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace weft::x86 {

// Select a register permutation for an in-bounds lookup of an SSA snapshot.
// An empty result means this table needs the addressable snapshot path.
mlir::Value registerLookup(mlir::ModuleOp module, mlir::OpBuilder &builder,
                           mlir::Location loc, mlir::Value table,
                           mlir::Value indices, unsigned vectorBits);

} // namespace weft::x86
