#pragma once

#include "Weft/Target/RISCV/RISCVCompiler.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

namespace weft::riscv_internal {
std::vector<RISCVOperationDiagnostic> collectPhysicalOperations(riscv::KernelOp kernel);
}
