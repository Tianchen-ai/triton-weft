#ifndef WEFT_X86_INDEXED_VALUES_H
#define WEFT_X86_INDEXED_VALUES_H

#include "mlir/IR/Value.h"
#include <cstdint>

namespace weft::x86 {
// Equality of predicate lane expressions, without assuming runtime values.
bool hasUniformPredicateGroups(mlir::Value predicate, int64_t group);
}
#endif
