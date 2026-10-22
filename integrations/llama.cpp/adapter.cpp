#include "weft/ggml.h"
#include "weft/runtime.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

struct ggml_weft_ime1_binding {
  void (*selected[2][2])(const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t, size_t, size_t) = {};
  ggml_weft_ime1_t original = nullptr;
  uint64_t calls = 0;
};

struct ggml_weft_ime2_binding {
  void (*selected[2])(const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t, size_t, size_t) = {};
  ggml_weft_ime1_t original = nullptr;
  uint64_t calls = 0;
};

namespace {
using Dot = void (*)(const uint8_t *, const uint8_t *, float *, size_t);
using PackedDot = void (*)(const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t, size_t, size_t);

struct StorageContract {
  ggml_type weight;
  const char *weightEncoding;
  const char *activationEncoding;
};

constexpr StorageContract contracts[] = {
    {GGML_TYPE_Q4_0, "Q4_0", "Q8_0"}, {GGML_TYPE_Q4_1, "Q4_1", "Q8_1"},
    {GGML_TYPE_Q5_0, "Q5_0", "Q8_0"}, {GGML_TYPE_Q5_1, "Q5_1", "Q8_1"},
    {GGML_TYPE_Q8_0, "Q8_0", "Q8_0"}, {GGML_TYPE_Q2_K, "Q2_K", "Q8_K"},
    {GGML_TYPE_Q3_K, "Q3_K", "Q8_K"}, {GGML_TYPE_Q4_K, "Q4_K", "Q8_K"},
    {GGML_TYPE_Q5_K, "Q5_K", "Q8_K"}, {GGML_TYPE_Q6_K, "Q6_K", "Q8_K"}};

[[noreturn]] void fail(const std::string &message) {
  GGML_ABORT("Weft CPU integration: %s", message.c_str());
}

void require(bool condition, const std::string &message) {
  if (!condition) fail(message);
}

void require(bool condition, const char *message) {
  if (!condition) fail(message);
}

struct AtomicCounters {
  std::atomic<uint64_t> nodes{0}, extra{0}, calls{0}, quantizes{0}, elements{0}, weftQuantizes{0};
  ggml_weft_counters snapshot() const {
    const auto dotCalls = calls.load(std::memory_order_relaxed);
    return {nodes.load(std::memory_order_relaxed), extra.load(std::memory_order_relaxed),
             dotCalls, dotCalls, quantizes.load(std::memory_order_relaxed),
             elements.load(std::memory_order_relaxed), weftQuantizes.load(std::memory_order_relaxed)};
  }
};

struct alignas(64) ThreadCounters {
  std::array<AtomicCounters, GGML_TYPE_COUNT> types;
};

void accumulate(ggml_weft_counters &into, const ggml_weft_counters &value) {
  into.nodes += value.nodes;
  into.extra_buffer_nodes += value.extra_buffer_nodes;
  into.vec_dot_calls += value.vec_dot_calls;
  into.dot_products += value.dot_products;
  into.quantize_calls += value.quantize_calls;
  into.quantized_elements += value.quantized_elements;
  into.weft_quantize_calls += value.weft_quantize_calls;
}

struct CounterRegistry {
  std::mutex mutex;
  std::array<ggml_weft_counters, GGML_TYPE_COUNT> retired{};
  std::vector<ThreadCounters *> active;
  void attach(ThreadCounters *counters) {
    std::lock_guard<std::mutex> lock(mutex);
    active.push_back(counters);
  }
  void detach(ThreadCounters *counters) {
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t type = 0; type < retired.size(); ++type) accumulate(retired[type], counters->types[type].snapshot());
    for (auto it = active.begin(); it != active.end(); ++it) {
      if (*it == counters) { active.erase(it); return; }
    }
    fail("worker counters were not registered");
  }
  ggml_weft_counters snapshot(ggml_type type) {
    std::lock_guard<std::mutex> lock(mutex);
    auto result = retired[type];
    for (const auto *counters : active) accumulate(result, counters->types[type].snapshot());
    return result;
  }
};

struct ThreadState {
  std::shared_ptr<CounterRegistry> registry;
  ThreadCounters counters;
  ggml_type type = GGML_TYPE_COUNT;
  bool reference = false;
  bool stableGraph = false;
  std::array<std::array<bool, 2>, GGML_TYPE_COUNT> checkedAMX{};
  std::array<bool, GGML_TYPE_COUNT> checkedDots{}, checkedQuantizers{};
  std::array<std::array<bool, 2>, 2> checkedPacked{};
  std::array<bool, 2> checkedIME2{};
  void (*quantize)(const float *, uint8_t *, size_t) = nullptr;
  ~ThreadState() { if (registry) registry->detach(&counters); }
};
thread_local ThreadState worker;
struct DotDispatch {
  Dot function = nullptr;
  AtomicCounters *counters = nullptr;
  uint64_t calls = 0;
};
thread_local DotDispatch dotDispatch;
thread_local ggml_weft_ime1_binding packedDispatch;
thread_local ggml_weft_ime2_binding ime2Dispatch;
thread_local uint64_t amxDispatchCalls = 0;
thread_local std::vector<int16_t> amxScratch;
thread_local std::vector<float> amxScales;

struct Registration {
  weft::kernel kernel;
  Dot function;
  size_t block, weightBytes, activationBytes;
  std::string name;
  Registration(weft::kernel value, const StorageContract &contract) : kernel(std::move(value)) {
    auto info = kernel.info();
    require(info.argument_count == 3 && info.dimension_count == 1 && std::strcmp(info.dimensions[0], "K") == 0,
            "vec-dot requires three Views and exactly one runtime K");
    const auto *traits = ggml_get_type_traits_cpu(contract.weight);
    require(traits && traits->vec_dot && traits->nrows == 1, "upstream vec-dot row contract differs from the registered ABI");
    auto activation = traits->vec_dot_type;
    block = ggml_blck_size(contract.weight);
    weightBytes = ggml_type_size(contract.weight);
    activationBytes = ggml_type_size(activation);
    require(block == static_cast<size_t>(ggml_blck_size(activation)), "activation block geometry differs from the compiled dot");
    for (size_t index = 0; index < 3; ++index) {
      auto arg = kernel.argument(index);
      const char *encoding = index == 0 ? contract.weightEncoding : index == 1 ? contract.activationEncoding : "dense.f32";
      require(arg.kind == WEFT_VIEW && !arg.strided && arg.interleave_rows == 0 && arg.rank == 1 &&
              std::strcmp(arg.shape[0], index == 2 ? "1" : "K") == 0 && std::strcmp(arg.encoding, encoding) == 0 &&
              arg.access == (index == 2 ? WEFT_WRITE : WEFT_READ) &&
              arg.record_elements == (index == 2 ? 1 : block) &&
              arg.storage_bytes == (index == 0 ? weightBytes : index == 1 ? activationBytes : sizeof(float)) &&
              arg.alignment <= (index == 2 || (index == 1 && activation == GGML_TYPE_Q8_K) ? alignof(float) : 2),
              "artifact View/storage ABI differs from the actual GGML type " + std::string(ggml_type_name(contract.weight)));
    }
    function = reinterpret_cast<Dot>(kernel.entry());
    name = info.symbol;
  }
};

struct Quantizer {
  weft::kernel kernel;
  void (*function)(const float *, uint8_t *, size_t);
  explicit Quantizer(weft::kernel value, ggml_type type) : kernel(std::move(value)) {
    auto info = kernel.info();
    require(info.argument_count == 2 && info.dimension_count == 1 && std::strcmp(info.dimensions[0], "K") == 0,
            "activation quantization requires X[K], Y[K] and one runtime K");
    const char *encoding = type == GGML_TYPE_Q8_0 ? "Q8_0" : "Q8_K";
    for (size_t index = 0; index < 2; ++index) {
      auto arg = kernel.argument(index);
      require(arg.kind == WEFT_VIEW && !arg.strided && arg.interleave_rows == 0 && arg.rank == 1 &&
              std::strcmp(arg.shape[0], "K") == 0 &&
              std::strcmp(arg.encoding, index ? encoding : "dense.f32") == 0 &&
              arg.access == (index ? WEFT_WRITE : WEFT_READ) &&
              arg.record_elements == (index ? static_cast<size_t>(ggml_blck_size(type)) : 1) &&
              arg.storage_bytes == (index ? ggml_type_size(type) : sizeof(float)) &&
              arg.alignment <= (index && type == GGML_TYPE_Q8_0 ? 2 : alignof(float)),
              "activation quantizer artifact does not match the GGML row storage ABI");
    }
    function = reinterpret_cast<decltype(function)>(kernel.entry());
  }
};

struct PackedIME {
  weft::kernel kernel;
  PackedDot function;
  PackedIME(weft::kernel value, bool affine, size_t rows) : kernel(std::move(value)) {
    auto info = kernel.info();
    require(info.argument_count == 3 && info.dimension_count == 3 &&
            std::strcmp(info.dimensions[0], "N") == 0 && std::strcmp(info.dimensions[1], "K") == 0 &&
            std::strcmp(info.dimensions[2], "M") == 0, "IME row tile requires N, K, M dimensions");
    const char *encodings[] = {affine ? "IMEQ4Affine_I[16]" : "IMEQ4Centered_I[16]",
                              rows == 4 ? "IMEQ8_I[4]" : "IMEQ8_I[1]", "dense.f32"};
    for (size_t index = 0; index < 3; ++index) {
      auto arg = kernel.argument(index);
      require(arg.kind == WEFT_VIEW && arg.rank == 2 && arg.strided == (index == 2) &&
              std::strcmp(arg.encoding, encodings[index]) == 0 &&
              std::strcmp(arg.shape[0], index ? "M" : "N") == 0 &&
              std::strcmp(arg.shape[1], index == 2 ? "N" : "K") == 0 &&
              arg.access == (index == 2 ? WEFT_WRITE : WEFT_READ) &&
              arg.interleave_rows == (index == 0 ? 16 : index == 1 ? rows : 0) &&
              arg.record_elements == (index == 2 ? 1 : 32) &&
              arg.storage_bytes == (index == 0 ? (affine ? 19 : 18) : index == 1 ? 36 : 4) &&
              arg.alignment <= (index == 0 ? (affine ? 1 : 2) : 4),
              "IME artifact differs from the native packed row-tile ABI");
    }
    function = reinterpret_cast<decltype(function)>(kernel.entry());
  }
};

struct PackedIME2 {
  weft::kernel kernel;
  PackedDot function;
  PackedIME2(weft::kernel value, size_t rows) : kernel(std::move(value)) {
    auto info = kernel.info();
    require(info.argument_count == 3 && info.dimension_count == 3 &&
            std::strcmp(info.dimensions[0], "N") == 0 && std::strcmp(info.dimensions[1], "K") == 0 &&
            std::strcmp(info.dimensions[2], "M") == 0, "IME2 row tile requires N, K, M dimensions");
    const char *encodings[] = {"IME2Weight8Rows[32]",
                              rows == 4 ? "IME2Activation8Rows[4]" : "IME2Activation8Rows[1]", "dense.f32"};
    for (size_t index = 0; index < 3; ++index) {
      auto arg = kernel.argument(index);
      require(arg.kind == WEFT_VIEW && arg.rank == 2 && arg.strided == (index == 2) &&
              std::strcmp(arg.encoding, encodings[index]) == 0 &&
              std::strcmp(arg.shape[0], index ? "M" : "N") == 0 &&
              std::strcmp(arg.shape[1], index == 2 ? "N" : "K") == 0 &&
              arg.access == (index == 2 ? WEFT_WRITE : WEFT_READ) &&
              arg.interleave_rows == (index == 0 ? 32 : index == 1 ? rows : 0) &&
              arg.record_elements == (index == 2 ? 1 : 32) &&
              arg.storage_bytes == (index == 0 ? 34 : index == 1 ? 38 : 4) &&
              arg.alignment <= (index == 2 ? 4 : 2),
              "IME2 artifact differs from the native signed i8 block32 row-tile ABI");
    }
    function = reinterpret_cast<decltype(function)>(kernel.entry());
  }
};

struct PackedAMX {
  weft::kernel kernel;
  void (*function)(const uint8_t *, const uint8_t *, float *, size_t, size_t) = nullptr;
  PackedDot matrixFunction = nullptr;
  void (*fixedMatrix)(const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t) = nullptr;
  void (*scratchFunction)(const uint8_t *, const uint8_t *, float *, int16_t *, size_t, size_t) = nullptr;
  void (*scratchMatrix)(const uint8_t *, const uint8_t *, float *, uint8_t *, size_t, size_t, size_t, size_t, size_t) = nullptr;
  bool tileState = false;
  PackedAMX(weft::kernel value, ggml_type weightType, bool prefill) : kernel(std::move(value)) {
    const bool kBlock = weightType == GGML_TYPE_Q4_K || weightType == GGML_TYPE_Q6_K;
    require(weightType == GGML_TYPE_Q4_0 || kBlock, "unsupported packed AMX storage role");
    const size_t block = kBlock ? 256 : 32;
    const size_t weightBytes = weightType == GGML_TYPE_Q6_K ? 210 : kBlock ? 148 : 18;
    const size_t activationBytes = kBlock ? 292 : 34;
    const ggml_type activationType = kBlock ? GGML_TYPE_Q8_K : GGML_TYPE_Q8_0;
    auto info = kernel.info();
    bool fixed = prefill && info.dimension_count == 1 && std::strcmp(info.dimensions[0], "K") == 0;
    bool scratch = !prefill && !kBlock && info.argument_count == 4;
    bool matrixScratch = prefill && !kBlock && !fixed && info.argument_count == 4;
    require((info.argument_count == 3 || scratch || matrixScratch) && (fixed || (info.dimension_count == (prefill ? 3 : 2) &&
            std::strcmp(info.dimensions[0], "N") == 0 && std::strcmp(info.dimensions[1], "K") == 0 &&
            (!prefill || std::strcmp(info.dimensions[2], "M") == 0))),
            "AMX packed row tile requires W, Xq, Y and N,K[,M] dimensions");
    tileState = std::strstr(info.march, "+amx-") != nullptr;
    require(std::strncmp(info.march, "x86-64-", 7) == 0 && (prefill || !tileState),
            "AMX packed row tiles require x86 SIMD or an integer AMX prefill artifact");
    const char *encodings[] = {weightType == GGML_TYPE_Q6_K ? "AMXQ6_KRows[16]" : kBlock ? "AMXQ4_KRows[16]" : "AMXQ4_0Rows[16]",
                              kBlock ? "Q8_K" : "Q8_0", "dense.f32"};
    for (size_t index = 0; index < 3; ++index) {
      auto arg = kernel.argument(index);
      require(arg.kind == WEFT_VIEW && arg.strided == (prefill && index == 2) &&
              arg.rank == (prefill || index == 0 ? 2 : 1) &&
              std::strcmp(arg.encoding, encodings[index]) == 0 &&
              std::strcmp(arg.shape[0], fixed ? "32" : index == 0 ? "N" : prefill ? "M" : index == 1 ? "K" : "N") == 0 &&
              (!(prefill || index == 0) || std::strcmp(arg.shape[1], index == 2 ? (fixed ? "32" : "N") : "K") == 0) &&
              arg.access == (index == 2 ? WEFT_WRITE : WEFT_READ) &&
              arg.interleave_rows == (index == 0 ? 16 : 0) &&
              arg.record_elements == (index == 2 ? 1 : block) &&
              arg.storage_bytes == (index == 0 ? weightBytes : index == 1 ? activationBytes : 4) &&
              arg.alignment <= (index == 2 || (index == 1 && kBlock) ? 4 : 2),
              "artifact differs from the native AMX packed buffer ABI");
    }
    if (scratch) {
      auto arg = kernel.argument(3);
      require(arg.kind == WEFT_VIEW && !arg.strided && arg.rank == 1 &&
              std::strcmp(arg.encoding, "BlockSum32") == 0 && std::strcmp(arg.shape[0], "K") == 0 &&
              arg.access == WEFT_READWRITE && arg.interleave_rows == 0 &&
              arg.record_elements == 32 && arg.storage_bytes == 2 && arg.alignment <= 2,
              "AMX decode scratch differs from its per-record signed sum contract");
    }
    if (matrixScratch) {
      auto arg = kernel.argument(3);
      require(arg.kind == WEFT_VIEW && !arg.strided && arg.rank == 2 &&
              std::strcmp(arg.encoding, "BlockScale32Rows[32]") == 0 &&
              std::strcmp(arg.shape[0], "M") == 0 && std::strcmp(arg.shape[1], "K") == 0 &&
              arg.access == WEFT_READWRITE && arg.interleave_rows == 32 &&
              arg.record_elements == 32 && arg.storage_bytes == 4 && arg.alignment <= 4,
              "AMX prefill scratch differs from its interleaved F32 scale contract");
    }
    require(ggml_blck_size(weightType) == static_cast<int64_t>(block) &&
            ggml_type_size(weightType) == (weightType == GGML_TYPE_Q6_K ? 210 : kBlock ? 144 : 18) &&
            ggml_blck_size(activationType) == static_cast<int64_t>(block) && ggml_type_size(activationType) == activationBytes,
            "linked GGML differs from the registered AMX buffer contract");
    if (fixed) fixedMatrix = reinterpret_cast<decltype(fixedMatrix)>(kernel.entry());
    else if (matrixScratch) scratchMatrix = reinterpret_cast<decltype(scratchMatrix)>(kernel.entry());
    else if (prefill) matrixFunction = reinterpret_cast<PackedDot>(kernel.entry());
    else if (scratch) scratchFunction = reinterpret_cast<decltype(scratchFunction)>(kernel.entry());
    else function = reinterpret_cast<decltype(function)>(kernel.entry());
  }
};

// The explicit framework bindings name artifacts; the compiler sees only the
// ordinary typed author programs and never selects by these role names.
struct Adapter {
  weft::context context;
  weft::package package;
  std::array<std::unique_ptr<Registration>, GGML_TYPE_COUNT> entries;
  std::array<std::unique_ptr<Quantizer>, GGML_TYPE_COUNT> quantizers;
  std::array<std::array<std::unique_ptr<PackedIME>, 2>, 2> packed;
  std::atomic<uint64_t> packedCalls{0};
  std::array<std::unique_ptr<PackedIME2>, 2> ime2;
  std::atomic<uint64_t> ime2Calls{0};
  std::array<std::array<std::unique_ptr<PackedAMX>, 2>, GGML_TYPE_COUNT> amx;
  std::atomic<uint64_t> amxCalls{0};
  std::shared_ptr<CounterRegistry> counters = std::make_shared<CounterRegistry>();
  explicit Adapter(const char *directory) : package(context, directory) {
    auto bindings = [&](const std::string &group) {
      return package.has_binding_group(group) ? package.bindings(group) :
          std::vector<std::pair<std::string, std::string>>{};
    };
    for (const auto &[typeName, name] : bindings("ggml.vec_dot")) {
      bool matched = false;
      for (const auto &contract : contracts) {
        if (typeName != ggml_type_name(contract.weight)) continue;
        require(!entries[contract.weight], "duplicate registered weight type");
        entries[contract.weight] = std::make_unique<Registration>(package.get_kernel(name), contract);
        matched = true;
        break;
      }
      require(matched, "package requests an unsupported GGML vec-dot storage contract: " + typeName);
    }
    for (const auto &[typeName, name] : bindings("ggml.from_float")) {
      const auto type = typeName == "q8_0" ? GGML_TYPE_Q8_0 :
                        typeName == "q8_K" ? GGML_TYPE_Q8_K : GGML_TYPE_COUNT;
      require(type != GGML_TYPE_COUNT, "unsupported activation quantizer: " + typeName);
      require(!quantizers[type], "duplicate activation quantizer");
      quantizers[type] = std::make_unique<Quantizer>(package.get_kernel(name), type);
    }
    for (const auto &[role, name] : bindings("ggml.ime1")) {
      require(role == "q4_centered_prefill" || role == "q4_affine_prefill" ||
              role == "q4_centered_decode" || role == "q4_affine_decode", "unsupported IME packed role: " + role);
      const bool affine = role == "q4_affine_prefill" || role == "q4_affine_decode";
      const bool single = role == "q4_centered_decode" || role == "q4_affine_decode";
      require(!packed[affine][single], "duplicate IME packed role");
      packed[affine][single] = std::make_unique<PackedIME>(package.get_kernel(name), affine, single ? 1 : 4);
    }
    for (const auto &[role, name] : bindings("ggml.ime2")) {
      require(role == "i8_prefill" || role == "i8_decode", "unsupported IME2 packed role: " + role);
      const size_t single = role == "i8_decode";
      require(!ime2[single], "duplicate IME2 packed role");
      ime2[single] = std::make_unique<PackedIME2>(package.get_kernel(name), single ? 1 : 4);
    }
    for (const auto &[role, name] : bindings("ggml.amx")) {
      require(role == "q4_0_decode" || role == "q4_0_prefill" || role == "q4_K_decode" || role == "q4_K_prefill" || role == "q6_K_decode" || role == "q6_K_prefill", "unsupported AMX packed role: " + role);
      const bool prefill = role == "q4_0_prefill" || role == "q4_K_prefill" || role == "q6_K_prefill";
      const auto type = role == "q6_K_decode" || role == "q6_K_prefill" ? GGML_TYPE_Q6_K :
                        role == "q4_K_decode" || role == "q4_K_prefill" ? GGML_TYPE_Q4_K : GGML_TYPE_Q4_0;
      require(!amx[type][prefill], "duplicate AMX packed role");
      amx[type][prefill] = std::make_unique<PackedAMX>(package.get_kernel(name), type, prefill);
    }
    bool any = false;
    for (const auto &entry : entries) any |= bool(entry);
    for (const auto &entry : quantizers) any |= bool(entry);
    for (const auto &roles : packed) for (const auto &entry : roles) any |= bool(entry);
    for (const auto &entry : ime2) any |= bool(entry);
    for (const auto &roles : amx) for (const auto &entry : roles) any |= bool(entry);
    require(any, "package does not contain supported GGML bindings");
  }
};

Adapter &adapter() {
  static std::unique_ptr<Adapter> result = [] {
    const char *directory = std::getenv("WEFT_GGML_PACKAGE");
    require(directory && *directory, "GGML_WEFT requires an explicit WEFT_GGML_PACKAGE directory");
    try { return std::make_unique<Adapter>(directory); }
    catch (const std::exception &error) { fail(error.what()); }
  }();
  return *result;
}

ThreadState &threadState() {
  if (!worker.registry) {
    worker.registry = adapter().counters;
    worker.registry->attach(&worker.counters);
  }
  return worker;
}

bool matrixNode(const ggml_tensor *tensor) {
  return tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID;
}
}

extern "C" {
void ggml_weft_initialize(void) { (void)adapter(); }

void ggml_weft_begin_graph(int nodeCount, ggml_tensor *const *nodes, bool externalCallback) {
  auto &state = threadState();
  state.checkedDots.fill(false);
  state.checkedQuantizers.fill(false);
  for (auto &roles : state.checkedPacked) roles.fill(false);
  state.checkedIME2.fill(false);
  for (auto &roles : state.checkedAMX) roles.fill(false);
  state.stableGraph = !externalCallback;
  // Built-in GGML operations preserve vector permission. Its native IME path
  // may narrow affinity, which preserves earlier checks. Host callbacks can
  // change either fact, so their graphs retain checks at every matrix node.
  for (int i = 0; state.stableGraph && i < nodeCount; ++i)
    switch (nodes[i]->op) {
    case GGML_OP_MAP_CUSTOM1:
    case GGML_OP_MAP_CUSTOM2:
    case GGML_OP_MAP_CUSTOM3:
    case GGML_OP_CUSTOM:
      state.stableGraph = false;
      break;
    default: break;
    }
}

uint64_t ggml_weft_ime1_calls(void) { return adapter().packedCalls.load(std::memory_order_relaxed); }
uint64_t ggml_weft_ime2_calls(void) { return adapter().ime2Calls.load(std::memory_order_relaxed); }
uint64_t ggml_weft_amx_calls(void) { return adapter().amxCalls.load(std::memory_order_relaxed); }

ggml_weft_amx_binding ggml_weft_select_amx(ggml_type weightType, bool prefill, size_t k) {
  if (weightType != GGML_TYPE_Q4_0 && weightType != GGML_TYPE_Q4_K && weightType != GGML_TYPE_Q6_K) return {};
  const auto &entry = adapter().amx[weightType][prefill];
  auto &state = threadState();
  if (!entry || state.reference) return {};
  if (!state.stableGraph || !state.checkedAMX[weightType][prefill]) {
    try { entry->kernel.check_thread(); }
    catch (const std::exception &error) { fail(error.what()); }
    state.checkedAMX[weightType][prefill] = true;
  }
  if (entry->scratchFunction) {
    require(k % 32 == 0, "AMX decode scratch requires complete activation records");
    try { if (amxScratch.size() < k / 32) amxScratch.resize(k / 32); }
    catch (const std::exception &error) { fail(error.what()); }
  }
  uint8_t *matrixScratch = nullptr;
  if (entry->scratchMatrix) {
    require(k % 32 == 0, "AMX prefill scratch requires complete activation records");
    // A framework input-row block has 32 rows: K/32 F32 scales per row.
    // Storage is reused by this worker; the source fills it inside each call.
    try { if (amxScales.size() < k + 16) amxScales.resize(k + 16); }
    catch (const std::exception &error) { fail(error.what()); }
    matrixScratch = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(amxScales.data()) + 63) & ~uintptr_t(63));
  }
  return {entry->function, entry->matrixFunction, entry->fixedMatrix, entry->scratchFunction,
          entry->scratchFunction ? amxScratch.data() : nullptr, &amxDispatchCalls, entry->tileState,
          entry->scratchMatrix, matrixScratch};
}

size_t ggml_weft_call_ime1(ggml_weft_ime1_binding *binding, size_t block,
                        const uint8_t *x, const uint8_t *w, const uint8_t *zero,
                        float *y, size_t rows, size_t columns, size_t blocks, size_t stride) {
  const size_t tileRows = rows >= 4 ? 4 : 1;
  const auto selected = binding->selected[zero != nullptr][tileRows == 1];
  if (selected && block == 32 && rows > 0 && columns % 16 == 0) {
    selected(w, x, y, columns, blocks * 32, tileRows, stride, 1);
    ++binding->calls;
    return tileRows;
  }
  return binding->original(block, x, w, zero, y, rows, columns, blocks, stride);
}

ggml_weft_ime1_binding *ggml_weft_select_ime1(ggml_weft_ime1_t original, ggml_type weightType) {
  require(weightType == GGML_TYPE_Q4_0 || weightType == GGML_TYPE_Q4_1 || weightType == GGML_TYPE_Q4_K,
          "IME selection requires a supported packed weight type");
  size_t role = weightType != GGML_TYPE_Q4_0;
  const auto &entries = adapter().packed[role];
  if (!entries[0] && !entries[1]) return nullptr;
  auto &state = threadState();
  auto &binding = packedDispatch;
  for (auto &slots : binding.selected) for (auto &selected : slots) selected = nullptr;
  for (size_t single = 0; single < entries.size(); ++single) {
    if (!entries[single]) continue;
    if (!state.stableGraph || !state.checkedPacked[role][single]) {
      try { entries[single]->kernel.bind_thread(); }
      catch (const std::exception &error) { fail(error.what()); }
      state.checkedPacked[role][single] = true;
    }
    binding.selected[role][single] = entries[single]->function;
  }
  require(original, "missing original IME implementation");
  binding.original = original;
  return &binding;
}

size_t ggml_weft_call_ime2(ggml_weft_ime2_binding *binding, size_t block,
                        const uint8_t *x, const uint8_t *w, const uint8_t *zero,
                        float *y, size_t rows, size_t columns, size_t blocks, size_t stride) {
  const size_t tileRows = rows >= 4 ? 4 : 1;
  const auto selected = binding->selected[tileRows == 1];
  if (selected && block == 32 && !zero && rows > 0 && columns > 0 && columns % 32 == 0) {
    selected(w, x, y, columns, blocks * 32, tileRows, stride, 1);
    ++binding->calls;
    return tileRows;
  }
  return binding->original(block, x, w, zero, y, rows, columns, blocks, stride);
}

ggml_weft_ime2_binding *ggml_weft_select_ime2(ggml_weft_ime1_t original, ggml_type weightType) {
  require(weightType == GGML_TYPE_Q6_K || weightType == GGML_TYPE_Q8_0,
          "IME2 signed i8 selection requires Q6_K or Q8_0 packed weights");
  const auto &entries = adapter().ime2;
  if (!entries[0] && !entries[1]) return nullptr;
  auto &state = threadState();
  auto &binding = ime2Dispatch;
  for (size_t single = 0; single < entries.size(); ++single) {
    binding.selected[single] = nullptr;
    if (!entries[single]) continue;
    if (!state.stableGraph || !state.checkedIME2[single]) {
      try { entries[single]->kernel.check_thread(); }
      catch (const std::exception &error) { fail(error.what()); }
      state.checkedIME2[single] = true;
    }
    binding.selected[single] = entries[single]->function;
  }
  require(original, "missing original IME2 implementation");
  binding.original = original;
  return &binding;
}

void ggml_weft_end_node(void) {
  if (dotDispatch.calls) {
    dotDispatch.counters->calls.fetch_add(dotDispatch.calls, std::memory_order_relaxed);
    dotDispatch.calls = 0;
  }
  if (packedDispatch.calls) {
    adapter().packedCalls.fetch_add(packedDispatch.calls, std::memory_order_relaxed);
    packedDispatch.calls = 0;
  }
  if (amxDispatchCalls) {
    adapter().amxCalls.fetch_add(amxDispatchCalls, std::memory_order_relaxed);
    amxDispatchCalls = 0;
  }
  if (ime2Dispatch.calls) {
    adapter().ime2Calls.fetch_add(ime2Dispatch.calls, std::memory_order_relaxed);
    ime2Dispatch.calls = 0;
  }
}

bool ggml_weft_registered(enum ggml_type type) {
  return type >= 0 && type < GGML_TYPE_COUNT && bool(adapter().entries[type]);
}

const char *ggml_weft_kernel_name(enum ggml_type type) {
  return ggml_weft_registered(type) ? adapter().entries[type]->name.c_str() : nullptr;
}

bool ggml_weft_counters_get(enum ggml_type type, ggml_weft_counters *result) {
  if (!result || type < 0 || type >= GGML_TYPE_COUNT) return false;
  *result = adapter().counters->snapshot(type);
  return true;
}

void ggml_weft_begin_node(const struct ggml_tensor *tensor, int ith, bool use_ref) {
  auto &state = threadState();
  state.reference = use_ref;
  state.type = matrixNode(tensor) ? tensor->src[0]->type : GGML_TYPE_COUNT;
  state.quantize = nullptr;
  dotDispatch.function = nullptr;
  if (state.type != GGML_TYPE_COUNT && !use_ref) {
    if (const auto &entry = adapter().entries[state.type]) {
      if (!state.stableGraph || !state.checkedDots[state.type]) {
        try { entry->kernel.check_thread(); }
        catch (const std::exception &error) { fail(error.what()); }
        state.checkedDots[state.type] = true;
      }
      dotDispatch.function = entry->function;
      dotDispatch.counters = &state.counters.types[state.type];
    }
    const auto activation = ggml_get_type_traits_cpu(state.type)->vec_dot_type;
    if (const auto &entry = adapter().quantizers[activation]) {
      if (!state.stableGraph || !state.checkedQuantizers[activation]) {
        try { entry->kernel.check_thread(); }
        catch (const std::exception &error) { fail(error.what()); }
        state.checkedQuantizers[activation] = true;
      }
      state.quantize = entry->function;
    }
  }
  if (state.type != GGML_TYPE_COUNT && ith == 0)
    state.counters.types[state.type].nodes.fetch_add(1, std::memory_order_relaxed);
}

void ggml_weft_extra_node(const struct ggml_tensor *tensor, int ith) {
  if (matrixNode(tensor) && ith == 0)
    threadState().counters.types[tensor->src[0]->type].extra.fetch_add(1, std::memory_order_relaxed);
}

ggml_weft_dot_binding ggml_weft_select_vec_dot(enum ggml_type type, ggml_vec_dot_t original) {
  auto &state = threadState();
  if (state.reference || !ggml_weft_registered(type)) return {original, nullptr, nullptr};
  require(state.type == type && dotDispatch.function, "vec-dot dispatch is outside its checked worker node");
  return {original, dotDispatch.function, &dotDispatch.calls};
}

void ggml_weft_from_float(ggml_from_float_t original, const float *x, void *y, int64_t n) {
  auto &state = threadState();
  require(state.type != GGML_TYPE_COUNT, "quantization hook is outside a matrix node");
  const auto activation = ggml_get_type_traits_cpu(state.type)->vec_dot_type;
  if (state.quantize)
    require(original == ggml_get_type_traits_cpu(activation)->from_float,
            "activation quantization call differs from its registered GGML row contract");
  ggml_weft_from_float_packed(activation, original, x, y, n);
}

void ggml_weft_from_float_packed(ggml_type activation, ggml_from_float_t original,
                                 const float *x, void *y, int64_t n) {
  require(original && n >= 0, "invalid upstream activation quantization call");
  auto &state = threadState();
  require(state.type != GGML_TYPE_COUNT, "quantization hook is outside a matrix node");
  require(activation == ggml_get_type_traits_cpu(state.type)->vec_dot_type &&
          n % ggml_blck_size(activation) == 0,
          "packed activation quantization differs from the checked worker node");
  if (state.quantize) {
    state.quantize(x, static_cast<uint8_t *>(y), n);
    state.counters.types[state.type].weftQuantizes.fetch_add(1, std::memory_order_relaxed);
  } else original(x, y, n);
  auto &count = state.counters.types[state.type];
  count.quantizes.fetch_add(1, std::memory_order_relaxed);
  count.elements.fetch_add(n, std::memory_order_relaxed);
}
}
