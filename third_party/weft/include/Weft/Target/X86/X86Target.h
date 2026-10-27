#ifndef WEFT_TARGET_X86TARGET_H
#define WEFT_TARGET_X86TARGET_H

#include "llvm/Support/JSON.h"
#include <map>
#include <string>
#include <vector>

namespace weft {
struct X86Target {
  std::string cpu;
  unsigned vectorBits;
  std::vector<std::string> features;
  std::string baseCPU() const { return cpu.substr(0, cpu.find('+')); }
  bool hasFeature(const std::string &name) const;
  bool hasByteDot() const { return hasFeature("avx512vnni") || hasFeature("avxvnni"); }
  unsigned vectorRegisters() const { return vectorBits == 512 ? 32 : 16; }
  unsigned predicateRegisters() const { return vectorBits == 512 ? 7 : 0; }
};
struct X86NativeTarget {
  X86Target target;
  std::vector<int> cpus;
  std::vector<std::string> availableFeatures;
};
bool parseX86Target(const std::string &cpu, const std::string &abi,
                    int64_t vlen, X86Target &result, std::string &error);
bool queryNativeX86Target(X86NativeTarget &result, std::string &error);
bool checkX86ThreadState(const X86Target &target, std::string &error);
llvm::json::Object x86TargetJSON(const X86Target &target);
} // namespace weft
#endif
