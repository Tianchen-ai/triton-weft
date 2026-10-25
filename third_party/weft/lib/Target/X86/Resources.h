#ifndef WEFT_X86_RESOURCES_H
#define WEFT_X86_RESOURCES_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>

namespace mlir::func { class FuncOp; }

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
  RegisterUsage storage(mlir::Type type) const;
  bool fits(RegisterUsage usage) const {
    return usage.vectors <= vectorRegisters && usage.predicates <= predicateRegisters;
  }
};
struct LoopResources {
  RegisterUsage peak, surrounding, scratch;
  uint64_t work = 0, loads = 0, stores = 0, conversions = 0;
  double issueUnits = 0, criticalPath = 0, carriedPath = 0;
  uint64_t readBytes = 0, writtenBytes = 0, permutations = 0, unpricedOperations = 0;
};
struct InstructionCost {
  double issues = 0, latency = 0;
  uint64_t readBytes = 0, writtenBytes = 0;
};
struct LoopExecutionCost {
  double period = 0, startup = 0;
  double total(uint64_t iterations) const {
    return iterations ? startup + period * iterations : 0;
  }
};
// One physical period includes loop control and its recurrence. A short domain
// additionally pays the initial dependency path; unknown domains retain both
// terms instead of assuming an arbitrary runtime trip count.
LoopExecutionCost loopExecutionCost(const LoopResources &resources);
// Structural instruction classes and native packet widths, not CPU-cycle
// predictions. An unknown operation makes the estimate incomplete.
std::optional<InstructionCost> instructionCost(mlir::Operation *op,
                                              const ResourceModel &model);
// Actual SSA residency, including values retained across enclosing regions.
LoopResources analyzeLoop(mlir::scf::ForOp loop, const ResourceModel &model);
RegisterUsage functionRegisterPeak(mlir::func::FuncOp function, const ResourceModel &model);
mlir::DictionaryAttr resourceAttributes(mlir::OpBuilder &b, const LoopResources &resources);
void updateArraySchedule(mlir::scf::ForOp loop, mlir::OpBuilder &b,
                         mlir::ArrayRef<mlir::NamedAttribute> fields);
} // namespace weft::x86
#endif
