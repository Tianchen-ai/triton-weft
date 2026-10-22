#include "Weft/Target/X86Target.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#if defined(__linux__)
#include <sched.h>
#if defined(__x86_64__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

namespace weft {
bool X86Target::hasFeature(const std::string &name) const {
  return std::find(features.begin(), features.end(), name) != features.end();
}

bool parseX86Target(const std::string &cpu, const std::string &abi,
                    int64_t vlen, X86Target &result, std::string &error) {
  auto base = cpu.substr(0, cpu.find('+'));
  if ((base != "x86-64-v3" && base != "x86-64-v4") || abi != "sysv" || vlen != 0) {
    error = "x86 requires an x86-64-v3/v4 profile with explicit +features, abi=sysv and vlen_bits=0 (RVV only)";
    return false;
  }
  result = {cpu, base == "x86-64-v4" ? 512u : 256u,
            {"sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "cx16", "sahf",
             "avx", "avx2", "bmi", "bmi2", "f16c", "fma", "lzcnt", "movbe", "xsave"}};
  if (base == "x86-64-v4")
    for (auto name : {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"})
      result.features.emplace_back(name);
  for (size_t begin = cpu.find('+'); begin != std::string::npos;) {
    size_t end = cpu.find('+', begin + 1);
    auto feature = cpu.substr(begin + 1, end == std::string::npos ? end : end - begin - 1);
    if ((feature != "avxvnni" && feature != "avx512vnni" &&
         feature != "avx512vbmi" && feature != "avx512vbmi2" &&
         feature != "avx512fp16" &&
         feature != "amx-tile" && feature != "amx-bf16" && feature != "amx-int8") ||
        (feature != "avxvnni" && base != "x86-64-v4") || result.hasFeature(feature)) {
      error = "unsupported, repeated or incompatible x86 feature: " + feature;
      return false;
    }
    result.features.push_back(feature);
    begin = end;
  }
  if ((result.hasFeature("amx-bf16") || result.hasFeature("amx-int8")) && !result.hasFeature("amx-tile")) {
    error = "AMX arithmetic requires the explicit amx-tile capability";
    return false;
  }
  return true;
}

bool checkX86ThreadState(const X86Target &target, std::string &error) {
#if defined(__linux__) && defined(__x86_64__)
  // LLVM's host probe combines CPUID with OSXSAVE/XGETBV. CPUID alone does not
  // authorize AVX or AVX-512 execution in this thread.
  auto available = llvm::sys::getHostCPUFeatures();
  for (const auto &name : target.features)
    if (!available.lookup(name)) {
      error = "x86 execution requires CPU and OS vector-state support for " + name;
      return false;
    }
  if (target.hasFeature("amx-tile")) {
    unsigned long permitted = 0;
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &permitted) != 0) {
      error = std::string("cannot query AMX tile-data permission: ") + strerror(errno);
      return false;
    }
    if (!(permitted & (1ul << 18)) &&
        syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18ul) != 0) {
      error = std::string("cannot enable AMX tile-data state on this execution thread: ") + strerror(errno);
      return false;
    }
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &permitted) != 0 || !(permitted & (1ul << 18))) {
      error = "AMX tile-data state remains unavailable on this execution thread";
      return false;
    }
  }
  return true;
#else
  error = "native x86 execution requires x86-64 Linux";
  return false;
#endif
}

bool queryNativeX86Target(X86NativeTarget &result, std::string &error) {
#if defined(__linux__) && defined(__x86_64__)
  cpu_set_t original;
  if (sched_getaffinity(0, sizeof(original), &original) != 0) {
    error = std::string("cannot query x86 affinity: ") + strerror(errno);
    return false;
  }
  X86Target v3, v4;
  parseX86Target("x86-64-v3", "sysv", 0, v3, error);
  parseX86Target("x86-64-v4", "sysv", 0, v4, error);
  bool allV4 = true, allVNNI = true, allAVXVNNI = true,
       allVBMI = true, allVBMI2 = true, allFP16 = true,
       allAMXTile = true, allAMXBF16 = true, allAMXInt8 = true, ok = true;
  result.cpus.clear();
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &original)) continue;
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    if (sched_setaffinity(0, sizeof(one), &one) != 0) {
      error = std::string("cannot inspect x86 CPU ") + std::to_string(cpu) + ": " + strerror(errno);
      ok = false;
      break;
    }
    if (!checkX86ThreadState(v3, error)) { ok = false; break; }
    std::string v4Error;
    allV4 &= checkX86ThreadState(v4, v4Error);
    auto features = llvm::sys::getHostCPUFeatures();
    allVNNI &= features.lookup("avx512vnni");
    allAVXVNNI &= features.lookup("avxvnni");
    allVBMI &= features.lookup("avx512vbmi");
    allVBMI2 &= features.lookup("avx512vbmi2");
    allFP16 &= features.lookup("avx512fp16");
    allAMXTile &= features.lookup("amx-tile");
    allAMXBF16 &= features.lookup("amx-bf16");
    allAMXInt8 &= features.lookup("amx-int8");
    result.cpus.push_back(cpu);
  }
  if (sched_setaffinity(0, sizeof(original), &original) != 0) {
    error = std::string("cannot restore discovery affinity: ") + strerror(errno);
    return false;
  }
  if (!ok) return false;
  if (result.cpus.empty()) { error = "x86 discovery has an empty CPU set"; return false; }
  result.target = allV4 ? v4 : v3;
  for (auto [name, supported] : {std::pair{"avxvnni", allAVXVNNI},
                                std::pair{"avx512vnni", allV4 && allVNNI},
                                std::pair{"avx512vbmi", allV4 && allVBMI},
                                std::pair{"avx512vbmi2", allV4 && allVBMI2},
                                std::pair{"avx512fp16", allV4 && allFP16}}) {
    if (!supported) continue;
    result.target.cpu += std::string("+") + name;
    result.target.features.emplace_back(name);
  }
  // Discovery reports optional matrix capabilities without acquiring tile
  // state or changing the default SIMD JIT profile. Execution of an explicit
  // AMX artifact obtains and checks permission on its actual worker.
  result.availableFeatures = result.target.features;
  if (allV4 && allAMXTile) result.availableFeatures.emplace_back("amx-tile");
  if (allV4 && allAMXTile && allAMXBF16) result.availableFeatures.emplace_back("amx-bf16");
  if (allV4 && allAMXTile && allAMXInt8) result.availableFeatures.emplace_back("amx-int8");
  return true;
#else
  error = "native x86 discovery requires x86-64 Linux";
  return false;
#endif
}

llvm::json::Object x86TargetJSON(const X86Target &target) {
  llvm::json::Array features;
  for (const auto &name : target.features) features.push_back(name);
  llvm::json::Array matrix;
  if (target.hasFeature("amx-bf16")) matrix.push_back("amx-bf16");
  if (target.hasFeature("amx-int8")) matrix.push_back("amx-int8");
  return llvm::json::Object{{"target", llvm::json::Object{{"arch", "x86_64"}, {"march", target.cpu},
            {"abi", "sysv"}, {"vlen_bits", 0}, {"vector_bits", target.vectorBits},
            {"llvm_cpu", target.baseCPU()}}},
          {"isa_extensions", std::move(features)}, {"matrix_extensions", std::move(matrix)},
          {"register_file", llvm::json::Object{{"vector_bits", target.vectorBits},
            {"vector_registers", target.vectorRegisters()}, {"predicate_registers", target.predicateRegisters()}}}};
}
} // namespace weft
