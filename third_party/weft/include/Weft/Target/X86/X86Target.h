#ifndef WEFT_TARGET_X86TARGET_H
#define WEFT_TARGET_X86TARGET_H

#include "llvm/Support/JSON.h"
#include <map>
#include <string>
#include <vector>

namespace weft {
struct OneDNNBrgemmInfo {
  bool available = false;
  int major = 0;
  int minor = 0;
  int patch = 0;
  std::string runtime;
  std::string root;
  std::string library;
  bool f32 = false;
  bool f16 = false;
  bool bf16 = false;
  bool s8 = false;
  bool u8s8 = false;
  std::string reason;
};

struct X86Target {
  std::string cpu;
  unsigned vectorBits;
  std::vector<std::string> features;
  std::vector<std::string> hostAvailableFeatures;
  std::string baseCPU() const { return cpu.substr(0, cpu.find('+')); }
  bool hasFeature(const std::string &name) const;
  bool hasHostFeature(const std::string &name) const;
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
// Caller has already checked hardware support on every CPU in its affinity.
bool checkX86ExecutionState(const X86Target &target, std::string &error);
const OneDNNBrgemmInfo &probeOneDNNBrgemm();
llvm::json::Object oneDNNBrgemmJSON(const OneDNNBrgemmInfo &info);
llvm::json::Object x86TargetJSON(const X86Target &target);
} // namespace weft
#endif
