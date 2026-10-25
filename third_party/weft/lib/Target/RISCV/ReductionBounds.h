#ifndef WEFT_RISCV_REDUCTION_BOUNDS_H
#define WEFT_RISCV_REDUCTION_BOUNDS_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

namespace weft::riscv_internal {
bool needsSavedReductionBounds(riscv::ReduceOp reduction);
}
#endif
