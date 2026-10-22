#ifndef WEFT_X86_RESOURCES_H
#define WEFT_X86_RESOURCES_H

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace weft::x86 {
struct RegisterUsage {
  uint64_t vectors = 0, predicates = 0;
  RegisterUsage &operator+=(RegisterUsage rhs) {
    vectors += rhs.vectors; predicates += rhs.predicates; return *this;
  }
  RegisterUsage &operator-=(RegisterUsage rhs) {
    vectors -= rhs.vectors; predicates -= rhs.predicates; return *this;
  }
};
struct ResourceModel {
  unsigned vectorBits, vectorRegisters, predicateRegisters;
  static ResourceModel get(mlir::Operation *op);
  RegisterUsage storage(mlir::Type type, unsigned scalarLanes = 0) const;
  bool fits(RegisterUsage usage) const {
    return usage.vectors <= vectorRegisters && usage.predicates <= predicateRegisters;
  }
};
struct LoopResources {
  RegisterUsage peak, surrounding, scratch;
  uint64_t work = 0, loads = 0, stores = 0, conversions = 0;
};
// Actual SSA residency, including values retained across enclosing regions.
// scalarLanes describes a candidate packetization of a scalar pointwise body.
LoopResources analyzeLoop(mlir::scf::ForOp loop, const ResourceModel &model,
                          unsigned scalarLanes = 0);
mlir::DictionaryAttr resourceAttributes(mlir::OpBuilder &b, const LoopResources &resources);
void updateArraySchedule(mlir::scf::ForOp loop, mlir::OpBuilder &b,
                         mlir::ArrayRef<mlir::NamedAttribute> fields);
} // namespace weft::x86
#endif
