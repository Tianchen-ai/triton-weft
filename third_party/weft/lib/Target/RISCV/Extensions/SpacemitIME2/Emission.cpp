#include "../../IME/FragmentEmission.h"

namespace weft::riscv_emission {
mlir::FailureOr<llvm::SmallVector<std::string>> emitSpacemitIME2MMA(
    riscv::IMEFragmentMMAOp operation, llvm::ArrayRef<std::string> lhs,
    llvm::ArrayRef<std::string> rhs, llvm::ArrayRef<std::string> accumulators,
    const FragmentEmission &emission) {
  llvm::SmallVector<std::string> outputs;
  for (const std::string &accumulator : accumulators) {
    outputs.push_back(operation.getAccumulatorPolicy() == "reuse"
                          ? accumulator : emission.fresh("ime_acc"));
    if (operation.getAccumulatorPolicy() == "copy")
      emission.line("vint32m2_t " + outputs.back() + " = " + accumulator + ";");
  }
  // LLVM's Xsmtvdot spelling encodes DT=11 (i8). On A100 the same base
  // instruction uses 8x16 operands and an 8x8 accumulator, as verified by
  // the selected fragment contract. Register allocation remains LLVM's.
  emission.line("__asm__ volatile(");
  emission.line("  \".option push\\n\\t.option arch, +xsmtvdot\\n\\t\"");
  emission.line("  \"vsetvli t0, zero, e8, m1, ta, ma\\n\\t\"");
  for (size_t row = 0; row < lhs.size(); ++row)
    for (size_t column = 0; column < rhs.size(); ++column)
      emission.line("  \"smt.vmadot %[acc" + std::to_string(row * rhs.size() + column) +
                    "], %[lhs" + std::to_string(row) + "], %[rhs" +
                    std::to_string(column) + "]\\n\\t\"");
  emission.line("  \".option pop\\n\\t\"");
  std::string outputsText;
  for (auto [index, output] : llvm::enumerate(outputs)) {
    if (index) outputsText += ", ";
    outputsText += "[acc" + std::to_string(index) + "] \"+&vr\"(" + output + ")";
  }
  emission.line("  : " + outputsText);
  std::string inputsText;
  auto append = [&](llvm::StringRef role, llvm::ArrayRef<std::string> parts) {
    for (auto [index, part] : llvm::enumerate(parts)) {
      if (!inputsText.empty()) inputsText += ", ";
      inputsText += "[" + role.str() + std::to_string(index) + "] \"vr\"(" + part + ")";
    }
  };
  append("lhs", lhs);
  append("rhs", rhs);
  emission.line("  : " + inputsText);
  std::string clobbers;
  for (mlir::Attribute attr : operation.getAsmClobbers()) {
    if (!clobbers.empty()) clobbers += ", ";
    clobbers += "\"" + mlir::cast<mlir::StringAttr>(attr).getValue().str() + "\"";
  }
  emission.line("  : " + clobbers + ");");
  return outputs;
}
} // namespace weft::riscv_emission
