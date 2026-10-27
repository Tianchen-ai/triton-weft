#pragma once
#include "Weft/Target/X86/X86Compiler.h"

namespace weft::x86 {
std::unique_ptr<mlir::Pass> createCanonicalConversionPass(
    const X86Target &target, llvm::StringRef tuneCPU);
}
