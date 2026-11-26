#include "Weft/Target/X86/X86Target.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

#if defined(__linux__) && defined(__x86_64__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(WEFT_ONEDNN_HEADER_AVAILABLE) && WEFT_ONEDNN_HEADER_AVAILABLE
#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_ukernel.h"
#endif

namespace weft {
namespace {

#if defined(__linux__) && defined(__x86_64__) && defined(WEFT_ONEDNN_HEADER_AVAILABLE) && \
    WEFT_ONEDNN_HEADER_AVAILABLE && defined(DNNL_EXPERIMENTAL_UKERNEL)

struct OneDNNAPI {
  const dnnl_version_t *(*version)() = nullptr;
  dnnl_status_t (*create)(dnnl_brgemm_t *, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t,
                          dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t,
                          dnnl_data_type_t, dnnl_data_type_t, dnnl_data_type_t) = nullptr;
  dnnl_status_t (*finalize)(dnnl_brgemm_t) = nullptr;
  dnnl_status_t (*generate)(dnnl_brgemm_t) = nullptr;
  dnnl_status_t (*get_scratchpad_size)(const_dnnl_brgemm_t, size_t *) = nullptr;
  dnnl_status_t (*set_hw_context)(const_dnnl_brgemm_t) = nullptr;
  dnnl_status_t (*execute)(const_dnnl_brgemm_t, const void *, const void *,
                           const dnnl_dim_t *, void *, void *) = nullptr;
  dnnl_status_t (*release_hw_context)() = nullptr;
  dnnl_status_t (*destroy)(dnnl_brgemm_t) = nullptr;
};

template <typename Fn>
bool loadSymbol(void *handle, const char *name, Fn &out) {
  dlerror();
  void *sym = dlsym(handle, name);
  if (dlerror() || !sym) return false;
  out = reinterpret_cast<Fn>(sym);
  return true;
}

bool ensureAMXPermission() {
  unsigned long permitted = 0;
  if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &permitted) != 0)
    return false;
  if (!(permitted & (1ul << 18)) &&
      syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18ul) != 0)
    return false;
  if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &permitted) != 0)
    return false;
  return (permitted & (1ul << 18)) != 0;
}

bool runProbeKernel(const OneDNNAPI &api, dnnl_data_type_t a_dt,
                    dnnl_data_type_t b_dt, dnnl_data_type_t c_dt,
                    dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K,
                    dnnl_dim_t lda, dnnl_dim_t ldb, dnnl_dim_t ldc,
                    const void *A, const void *B, void *C, bool releaseAMX) {
  dnnl_brgemm_t brg = nullptr;
  if (api.create(&brg, M, N, K, 1, lda, ldb, ldc, a_dt, b_dt, c_dt) != dnnl_success || !brg)
    return false;
  bool ok = false;
  if (api.finalize(brg) == dnnl_success && api.generate(brg) == dnnl_success) {
    size_t scratchBytes = 0;
    if (api.get_scratchpad_size(brg, &scratchBytes) == dnnl_success) {
      size_t allocBytes = ((scratchBytes + 63) / 64) * 64;
      if (allocBytes == 0) allocBytes = 64;
      void *scratch = nullptr;
      if (posix_memalign(&scratch, 64, allocBytes) == 0 && scratch) {
        std::memset(scratch, 0, allocBytes);
        dnnl_dim_t offsets[2] = {0, 0};
        if (api.set_hw_context(brg) == dnnl_success) {
          if (api.execute(brg, A, B, offsets, C, scratch) == dnnl_success)
            ok = true;
          if (releaseAMX) api.release_hw_context();
        }
        std::free(scratch);
      }
    }
  }
  api.destroy(brg);
  return ok;
}

#endif

OneDNNBrgemmInfo runOneDNNProbe() {
  OneDNNBrgemmInfo info;
#if defined(__linux__) && defined(__x86_64__) && defined(WEFT_ONEDNN_HEADER_AVAILABLE) && \
    WEFT_ONEDNN_HEADER_AVAILABLE && defined(DNNL_EXPERIMENTAL_UKERNEL)
  const char *envRoot = std::getenv("WEFT_ONEDNN_ROOT");
  std::string root = (envRoot && *envRoot) ? std::string(envRoot) : std::string(WEFT_ONEDNN_ROOT_DIR);
  if (root.empty()) {
    info.reason = "set WEFT_ONEDNN_ROOT to a oneDNN ukernel installation";
    return info;
  }
  info.root = root;
  llvm::SmallString<256> headerPath(root);
  llvm::sys::path::append(headerPath, "include", "oneapi", "dnnl", "dnnl_ukernel.h");
  if (!llvm::sys::fs::exists(headerPath)) {
    info.reason = "oneDNN ukernel header not found under " + root;
    return info;
  }

  std::string libPath;
  for (const char *candidate : {"lib/libdnnl.so.3", "lib/libdnnl.so", "lib64/libdnnl.so.3", "lib64/libdnnl.so"}) {
    llvm::SmallString<256> path(root);
    llvm::sys::path::append(path, candidate);
    if (llvm::sys::fs::exists(path)) {
      libPath = std::string(path.str());
      break;
    }
  }
  if (libPath.empty() && !(envRoot && *envRoot) && std::strlen(WEFT_ONEDNN_LIBRARY_PATH) > 0 &&
      llvm::sys::fs::exists(WEFT_ONEDNN_LIBRARY_PATH)) {
    libPath = WEFT_ONEDNN_LIBRARY_PATH;
  }
  if (libPath.empty()) {
    info.reason = "libdnnl shared library not found under " + root;
    return info;
  }
  info.library = libPath;

  // Check build-time SEQ runtime configuration from the matched installation header.
  llvm::SmallString<256> configPath(root);
  llvm::sys::path::append(configPath, "include", "oneapi", "dnnl", "dnnl_config.h");
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SEQ
  info.runtime = "SEQ";
#else
  info.runtime = "NON_SEQ";
  info.reason = "oneDNN CPU runtime must be SEQ";
  return info;
#endif

  void *handle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char *err = dlerror();
    info.reason = std::string("dlopen failed for ") + libPath + (err ? std::string(": ") + err : "");
    return info;
  }

  OneDNNAPI api;
  bool symbolsOk =
      loadSymbol(handle, "dnnl_version", api.version) &&
      loadSymbol(handle, "dnnl_brgemm_create", api.create) &&
      loadSymbol(handle, "dnnl_brgemm_finalize", api.finalize) &&
      loadSymbol(handle, "dnnl_brgemm_generate", api.generate) &&
      loadSymbol(handle, "dnnl_brgemm_get_scratchpad_size", api.get_scratchpad_size) &&
      loadSymbol(handle, "dnnl_brgemm_set_hw_context", api.set_hw_context) &&
      loadSymbol(handle, "dnnl_brgemm_execute", api.execute) &&
      loadSymbol(handle, "dnnl_brgemm_release_hw_context", api.release_hw_context) &&
      loadSymbol(handle, "dnnl_brgemm_destroy", api.destroy);
  if (!symbolsOk) {
    dlclose(handle);
    info.reason = "libdnnl does not export complete experimental BRGeMM ukernel symbols";
    return info;
  }

  const dnnl_version_t *ver = api.version();
  if (!ver) {
    dlclose(handle);
    info.reason = "dnnl_version() returned null";
    return info;
  }
  info.major = ver->major;
  info.minor = ver->minor;
  info.patch = ver->patch;
  if (info.major < 3 || (info.major == 3 && info.minor < 13)) {
    dlclose(handle);
    info.reason = "oneDNN >= 3.13 is required for BRGeMM ukernels";
    return info;
  }

  auto hostFeatures = llvm::sys::getHostCPUFeatures();
  if (!hostFeatures.lookup("avx512f")) {
    dlclose(handle);
    info.reason = "host CPU lacks AVX-512F required for x86 BRGeMM ukernels";
    return info;
  }

  // Probe f32 BRGeMM ukernel execution.
  {
    alignas(64) float A[16];
    alignas(64) float B[64];
    alignas(64) float C[64];
    for (int i = 0; i < 16; ++i) A[i] = 2.0f;
    for (int i = 0; i < 64; ++i) { B[i] = 3.0f; C[i] = 0.0f; }
    if (runProbeKernel(api, dnnl_f32, dnnl_f32, dnnl_f32, 4, 16, 4, 4, 16, 16, A, B, C, false) &&
        std::fabs(C[0] - 24.0f) < 1e-4f) {
      info.f32 = true;
    }
  }

  // Probe f16 BRGeMM ukernel execution.
  if (hostFeatures.lookup("avx512fp16")) {
    alignas(64) uint16_t A[16];
    alignas(64) uint16_t B[64];
    alignas(64) float C[64];
    for (int i = 0; i < 16; ++i) A[i] = 0x4000u; // 2.0 in fp16
    for (int i = 0; i < 64; ++i) { B[i] = 0x4200u; C[i] = 0.0f; } // 3.0 in fp16
    if (runProbeKernel(api, dnnl_f16, dnnl_f16, dnnl_f32, 4, 16, 4, 4, 16, 16, A, B, C, false) &&
        std::fabs(C[0] - 24.0f) < 1e-3f) {
      info.f16 = true;
    }
  }

  // Probe AMX BF16 / S8 / U8S8 BRGeMM ukernel execution when OS permits tile state.
  if (hostFeatures.lookup("amx-tile") && ensureAMXPermission()) {
    if (hostFeatures.lookup("amx-bf16")) {
      alignas(64) uint16_t A[16];
      alignas(64) uint16_t B[64]; // [K/2=2, ldb=16, 2]
      alignas(64) float C[64];
      for (int i = 0; i < 16; ++i) A[i] = 0x4000u; // 2.0 in bf16
      for (int i = 0; i < 64; ++i) { B[i] = 0x4040u; C[i] = 0.0f; } // 3.0 in bf16
      if (runProbeKernel(api, dnnl_bf16, dnnl_bf16, dnnl_f32, 4, 16, 4, 4, 16, 16, A, B, C, true) &&
          std::fabs(C[0] - 24.0f) < 1e-3f) {
        info.bf16 = true;
      }
    }
    if (hostFeatures.lookup("amx-int8")) {
      alignas(64) int8_t As[16];
      alignas(64) uint8_t Au[16];
      alignas(64) int8_t B[64]; // [K/4=1, ldb=16, 4]
      alignas(64) int32_t C[64];
      for (int i = 0; i < 16; ++i) { As[i] = -2; Au[i] = 2; }
      for (int i = 0; i < 64; ++i) { B[i] = 3; C[i] = 0; }
      if (runProbeKernel(api, dnnl_s8, dnnl_s8, dnnl_s32, 4, 16, 4, 4, 16, 16, As, B, C, true) &&
          C[0] == -24) {
        info.s8 = true;
      }
      std::memset(C, 0, sizeof(C));
      if (runProbeKernel(api, dnnl_u8, dnnl_s8, dnnl_s32, 4, 16, 4, 4, 16, 16, Au, B, C, true) &&
          C[0] == 24) {
        info.u8s8 = true;
      }
    }
  }

  info.available = info.f32;
  if (!info.available) {
    info.reason = "oneDNN BRGeMM f32 ukernel probe failed to execute";
  }
  // Keep libdnnl resident in process address space so repeated queries avoid
  // dlclose/dlopen churn on oneDNN's internal ISA singletons.
#else
  info.reason = "oneDNN BRGeMM ukernel support is only built on x86-64 Linux with oneDNN headers";
#endif
  return info;
}

} // namespace

const OneDNNBrgemmInfo &probeOneDNNBrgemm() {
  static std::once_flag once;
  static OneDNNBrgemmInfo cached;
  std::call_once(once, [] { cached = runOneDNNProbe(); });
  return cached;
}

llvm::json::Object oneDNNBrgemmJSON(const OneDNNBrgemmInfo &info) {
  llvm::json::Array dtypes;
  if (info.f32) dtypes.push_back("f32");
  if (info.f16) dtypes.push_back("f16");
  if (info.bf16) dtypes.push_back("bf16");
  if (info.s8) dtypes.push_back("s8");
  if (info.u8s8) dtypes.push_back("u8s8");
  return llvm::json::Object{
      {"available", info.available},
      {"version", llvm::json::Array{info.major, info.minor, info.patch}},
      {"runtime", info.runtime},
      {"root", info.root},
      {"library", info.library},
      {"dtypes", std::move(dtypes)},
      {"f32", info.f32},
      {"f16", info.f16},
      {"bf16", info.bf16},
      {"s8", info.s8},
      {"u8s8", info.u8s8},
      {"reason", info.reason}};
}

} // namespace weft
