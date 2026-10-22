#include "Internal.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <sched.h>
#include <set>
#include <sys/prctl.h>

namespace weft::runtime {

[[maybe_unused]] static llvm::json::Value cacheJSON(const std::optional<RISCVDataCache> &cache) {
  if (!cache) return nullptr;
  return llvm::json::Object{{"size_bytes", cache->sizeBytes}, {"line_bytes", cache->lineBytes},
      {"ways", cache->ways}, {"sets", cache->sets}};
}

Context::Context() {
  std::string error;
#if defined(__linux__) && defined(__x86_64__)
  require(queryNativeX86Target(x86, error), WEFT_TARGET_ERROR, error);
  CPU_ZERO(&allowedCPUs);
  llvm::json::Array cpus;
  for (int cpu : x86.cpus) { CPU_SET(cpu, &allowedCPUs); cpus.push_back(cpu); }
  auto facts = x86TargetJSON(x86.target);
  llvm::json::Array available;
  for (const auto &name : x86.availableFeatures) available.push_back(name);
  facts["available_isa_extensions"] = std::move(available);
  facts["cpus"] = std::move(cpus);
  json = llvm::formatv("{0}", llvm::json::Value(std::move(facts))).str();
#else
  require(queryNativeRISCVTarget(native, error), WEFT_TARGET_ERROR, error);
#if defined(__linux__)
  CPU_ZERO(&allowedCPUs);
  for (int cpu : native.cpus) CPU_SET(cpu, &allowedCPUs);
  for (const auto &[extension, cpus] : native.matrixExtensions) {
    auto &mask = extensionCPUs[extension];
    CPU_ZERO(&mask);
    for (int cpu : cpus) CPU_SET(cpu, &mask);
  }
#endif
  llvm::json::Array cpus;
  for (int cpu : native.cpus) cpus.push_back(cpu);
  llvm::json::Object isa, matrix;
  llvm::json::Array dataCaches;
  for (const auto &observation : native.dataCaches)
    dataCaches.push_back(llvm::json::Object{{"cpu", observation.cpu}, {"cache", cacheJSON(observation.cache)}});
  for (const auto &[name, version] : native.isaExtensions)
    isa[name] = llvm::json::Array{version.first, version.second};
  for (const auto &[name, supported] : native.matrixExtensions) {
    llvm::json::Array values;
    for (int cpu : supported) values.push_back(cpu);
    matrix[name] = std::move(values);
  }
  json = llvm::formatv("{0}", llvm::json::Value(llvm::json::Object{
      {"march", native.profile.march}, {"abi", native.profile.abi},
      {"xlen", native.profile.xlen}, {"vlen_bits", native.profile.vlenBits},
      {"data_cache", cacheJSON(native.profile.dataCache)}, {"cpu_data_caches", std::move(dataCaches)},
      {"cpus", std::move(cpus)}, {"isa_extensions", std::move(isa)},
      {"matrix_extensions", std::move(matrix)}})).str();
#endif
}

void Context::requireAvailable(const std::vector<std::string> &extensions) const {
#if defined(__linux__) && defined(__x86_64__)
  for (const auto &name : extensions)
    require((name == "amx-bf16" || name == "amx-int8") &&
            std::find(x86.availableFeatures.begin(), x86.availableFeatures.end(), name) != x86.availableFeatures.end(),
            WEFT_TARGET_ERROR, "x86 matrix extension is unavailable in the allowed CPU set: " + name);
#elif defined(__linux__) && defined(__riscv)
  cpu_set_t available = allowedCPUs;
  for (const auto &extension : extensions) {
    auto found = extensionCPUs.find(extension);
    if (found == extensionCPUs.end())
      throw Failure(WEFT_TARGET_ERROR, extension + " is unavailable in the discovered CPU set");
    CPU_AND(&available, &available, &found->second);
  }
  if (!CPU_COUNT(&available))
    throw Failure(WEFT_TARGET_ERROR, "artifact extensions have no common execution CPU");
#else
  (void)extensions;
  throw Failure(WEFT_TARGET_ERROR, "native execution requires RISC-V Linux");
#endif
}

void Context::check(const std::vector<std::string> &extensions) const {
  verifyThread(extensions, false);
}

void Context::bindThread(const std::vector<std::string> &extensions) const {
  verifyThread(extensions, true);
}

void Context::verifyThread(const std::vector<std::string> &extensions, bool bind) const {
#if defined(__linux__) && defined(__x86_64__)
  (void)bind;
  requireAvailable(extensions);
  cpu_set_t current, intersection;
  require(sched_getaffinity(0, sizeof(current), &current) == 0,
          WEFT_TARGET_ERROR, "cannot query x86 execution affinity");
  CPU_AND(&intersection, &current, &allowedCPUs);
  require(CPU_EQUAL(&intersection, &current), WEFT_TARGET_ERROR,
          "execution affinity exceeds the discovered CPU set");
  std::string error;
  X86Target executing = x86.target;
  if (!extensions.empty()) {
    executing.features.emplace_back("amx-tile");
    for (const auto &extension : extensions) executing.features.push_back(extension);
  }
  require(checkX86ThreadState(executing, error), WEFT_TARGET_ERROR, error);
#elif defined(__linux__) && defined(__riscv)
  int control = prctl(70, 0UL, 0UL, 0UL, 0UL);
  if (control < 0 || (control & 3) == 1)
    throw Failure(WEFT_TARGET_ERROR, "calling thread has no enabled RISC-V vector state");
  cpu_set_t current;
  if (sched_getaffinity(0, sizeof(current), &current) != 0)
    throw Failure(WEFT_TARGET_ERROR, std::string("cannot query execution affinity: ") + strerror(errno));
  cpu_set_t intersection;
  CPU_AND(&intersection, &current, &allowedCPUs);
  if (!CPU_EQUAL(&intersection, &current))
    throw Failure(WEFT_TARGET_ERROR, "execution affinity exceeds the discovered CPU set");
  cpu_set_t eligible = current;
  for (const auto &extension : extensions) {
    auto found = extensionCPUs.find(extension);
    if (found == extensionCPUs.end())
      throw Failure(WEFT_TARGET_ERROR, extension + " is unavailable on execution affinity");
    CPU_AND(&eligible, &eligible, &found->second);
    if (!bind && !CPU_EQUAL(&eligible, &current))
      throw Failure(WEFT_TARGET_ERROR, extension + " is unavailable on execution affinity");
  }
  if (bind && !CPU_EQUAL(&eligible, &current)) {
    if (!CPU_COUNT(&eligible))
      throw Failure(WEFT_TARGET_ERROR, "current affinity has no CPU supporting the artifact extensions");
    if (sched_setaffinity(0, sizeof(eligible), &eligible) != 0)
      throw Failure(WEFT_TARGET_ERROR, std::string("cannot bind extension execution affinity: ") + strerror(errno));
  }
#else
  (void)extensions;
  (void)bind;
  throw Failure(WEFT_TARGET_ERROR, "native execution requires RISC-V Linux");
#endif
}

static void *symbolAddress(void *library, const std::string &name) {
  dlerror();
  void *address = dlsym(library, name.c_str());
  const char *error = dlerror();
  require(!error && address, WEFT_ABI_ERROR, "artifact has no callable " + name +
          (error ? ": " + std::string(error) : ""));
  return address;
}

Kernel::Kernel(std::shared_ptr<Context> context, const std::string &directory)
    : Kernel(std::move(context), directory + "/kernel.so", readObject(directory + "/manifest.json")) {
  this->directory = directory;
}

Kernel::Kernel(std::shared_ptr<Context> context, const std::string &libraryPath,
               const llvm::json::Object &manifest)
    : context(std::move(context)) {
  require(string(manifest, "kind") == "weft-callable-artifact", WEFT_ABI_ERROR, "unsupported artifact kind");
  require(boolean(manifest, "native_library"), WEFT_UNSUPPORTED, "artifact has no native library; link it before deployment");
  auto dispatchSymbol = string(manifest, "dispatch");
  require(dispatchSymbol == "weft_kernel_dispatch" || dispatchSymbol == "triton_weft_instance_dispatch",
          WEFT_ABI_ERROR, "artifact has no typed native dispatch contract");
  const auto &metadata = object(manifest, "kernel");
  const auto &options = object(manifest, "options");
  symbol = string(metadata, "symbol");
  march = string(metadata, "march");
  abi = string(metadata, "abi");
  vlen = integer(metadata, "vlen_bits");
  auto arch = object(manifest, "target").getString("arch");
  if (arch && *arch == "x86_64") {
    X86Target requested;
    std::string diagnostic;
    require(parseX86Target(march, abi, vlen, requested, diagnostic), WEFT_ABI_ERROR, diagnostic);
    require(march == string(object(manifest, "target"), "march") &&
            abi == string(object(manifest, "target"), "abi") &&
            vlen == static_cast<size_t>(integer(object(manifest, "target"), "vlen_bits")),
            WEFT_ABI_ERROR, "kernel and artifact x86 target contracts disagree");
    require(!this->context->x86.cpus.empty() && sizeof(void *) == 8,
            WEFT_TARGET_ERROR, "x86 artifact requires an x86-64 process");
    require(requested.vectorBits <= this->context->x86.target.vectorBits, WEFT_TARGET_ERROR,
            "artifact requires a stronger x86 ISA than the allowed CPU set provides");
    const auto &declared = object(metadata, "isa_extensions");
    require(declared.size() == requested.features.size(), WEFT_ABI_ERROR,
            "artifact x86 feature requirements disagree with march");
    for (const auto &name : requested.features) {
      const auto &available = this->context->x86.availableFeatures;
      require(std::find(available.begin(), available.end(), name) != available.end(), WEFT_TARGET_ERROR,
              "artifact requires an x86 feature unavailable in the allowed CPU set: " + name);
      const auto *version = declared.getArray(name);
      require(version && version->size() == 2 && (*version)[0].getAsInteger() == 1 &&
              (*version)[1].getAsInteger() == 0, WEFT_ABI_ERROR, "invalid x86 requirement: " + name);
    }
    usedExtensions = strings(metadata, "used_extensions");
    auto matrix = strings(metadata, "matrix_extensions");
    std::vector<std::string> expectedMatrix;
    if (requested.hasFeature("amx-bf16")) expectedMatrix.emplace_back("amx-bf16");
    if (requested.hasFeature("amx-int8")) expectedMatrix.emplace_back("amx-int8");
    require(matrix == expectedMatrix, WEFT_ABI_ERROR, "x86 matrix capabilities disagree with march");
    for (const auto &name : usedExtensions)
      require(std::find(matrix.begin(), matrix.end(), name) != matrix.end(), WEFT_ABI_ERROR,
              "artifact uses an undeclared x86 matrix capability: " + name);
    require(strings(metadata, "required_extensions").empty() &&
            strings(options, "require_extensions").empty() && string(options, "matrix_extension") == "none",
            WEFT_UNSUPPORTED, "x86 matrix capabilities are selected through the explicit target feature set");
    xlen = 64;
  } else {
  RISCVTargetProfile requested;
  std::string diagnostic;
  auto matrixExtension = string(options, "matrix_extension");
  require(parseRISCVTargetProfile(march, abi, vlen, matrixExtension,
                                  integer(options, "max_widening_combine_groups"),
                                  string(options, "partial_combine_policy"),
                                  string(options, "record_axis_policy"), requested, diagnostic),
          WEFT_ABI_ERROR, diagnostic);
  xlen = requested.xlen;
  require(march == string(object(manifest, "target"), "march") &&
          abi == string(object(manifest, "target"), "abi") &&
          vlen == static_cast<size_t>(integer(object(manifest, "target"), "vlen_bits")),
          WEFT_ABI_ERROR, "kernel and artifact target contracts disagree");
  require(abi == this->context->native.profile.abi && vlen == static_cast<size_t>(this->context->native.profile.vlenBits) &&
          sizeof(void *) * 8 == xlen, WEFT_TARGET_ERROR, "artifact ABI or VLEN differs from the native process");
  std::map<std::string, std::pair<unsigned, unsigned>> expectedISA;
  require(parseRISCVISAExtensions(march, vlen, expectedISA, diagnostic), WEFT_ABI_ERROR, diagnostic);
  const auto &declaredISA = object(metadata, "isa_extensions");
  require(declaredISA.size() == expectedISA.size(), WEFT_ABI_ERROR, "artifact ISA requirements disagree with march");
  for (const auto &[name, version] : expectedISA) {
    const auto *pair = declaredISA.getArray(name);
    require(pair && pair->size() == 2 && (*pair)[0].getAsInteger() && (*pair)[1].getAsInteger() &&
            *(*pair)[0].getAsInteger() == version.first && *(*pair)[1].getAsInteger() == version.second,
            WEFT_ABI_ERROR, "inconsistent ISA requirement: " + name);
    auto available = this->context->native.isaExtensions.find(name);
    require(available != this->context->native.isaExtensions.end() && available->second >= version,
            WEFT_TARGET_ERROR, "native target does not provide required ISA extension " + name);
  }
  usedExtensions = strings(metadata, "used_extensions");
  auto capabilities = strings(metadata, "matrix_extensions");
  auto required = strings(metadata, "required_extensions");
  auto expectedCapabilities = matrixExtension == "none" ? std::vector<std::string>{} :
                              std::vector<std::string>{matrixExtension};
  require(capabilities == expectedCapabilities, WEFT_ABI_ERROR,
          "artifact capability binding differs from its compilation options");
  auto declaredRequirements = strings(options, "require_extensions");
  std::sort(required.begin(), required.end());
  std::sort(declaredRequirements.begin(), declaredRequirements.end());
  require(required == declaredRequirements, WEFT_ABI_ERROR,
          "artifact required extensions differ from its compilation options");
  auto tileStrategy = string(options, "local_tile_strategy");
  require(string(metadata, "local_tile_strategy") == tileStrategy, WEFT_ABI_ERROR,
          "artifact local tile strategy differs from its compilation options");
  if (tileStrategy != "target-default") {
    auto applied = object(metadata, "local_tile_realizations").getInteger(tileStrategy);
    require(applied && *applied > 0, WEFT_ABI_ERROR, "artifact did not apply its selected local tile strategy");
  }
  for (const auto &name : usedExtensions)
    require(std::find(capabilities.begin(), capabilities.end(), name) != capabilities.end(),
            WEFT_ABI_ERROR, "artifact uses an undeclared matrix capability: " + name);
  for (const auto &name : required)
    require(std::find(usedExtensions.begin(), usedExtensions.end(), name) != usedExtensions.end(),
            WEFT_ABI_ERROR, "artifact did not fulfill required extension " + name);
  }
  this->context->requireAvailable(usedExtensions);

  dimensions = strings(metadata, "shape_parameters");
  for (const auto &[name, value] : object(metadata, "bindings")) {
    auto number = value.getAsInteger();
    require(number && *number > 0, WEFT_ABI_ERROR, "source binding must be a positive integer");
    bindings.emplace(name.str(), *number);
  }
  for (const auto &[name, value] : object(options, "meta")) {
    auto number = value.getAsInteger();
    auto found = bindings.find(name.str());
    require(number && found != bindings.end() && *number > 0 && found->second == static_cast<size_t>(*number),
            WEFT_ABI_ERROR, "source binding differs from compiled ABI: " + name.str());
  }
  std::set<std::string> names;
  for (const auto &name : dimensions) {
    require(names.insert(name).second && !bindings.count(name), WEFT_ABI_ERROR, "duplicate or static runtime dimension: " + name);
    dimensionNames.push_back(name.c_str());
  }
  const auto &parameters = array(metadata, "arguments");
  arguments.reserve(parameters.size());
  for (const auto &parameter : parameters) {
    auto *record = parameter.getAsObject();
    require(record != nullptr, WEFT_ABI_ERROR, "argument metadata is not an object");
    arguments.emplace_back(*record);
    require(names.insert(arguments.back().name).second, WEFT_ABI_ERROR, "duplicate ABI parameter name");
  }
  for (auto &argument : arguments) {
    for (const auto &axis : argument.shape) {
      argument.shapeNames.push_back(axis.c_str());
    }
  }
  contract = std::make_unique<CallContract>(arguments, dimensions, bindings);
  void *handle = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) throw Failure(WEFT_IO_ERROR, "cannot load kernel library: " + std::string(dlerror()));
  try {
    entry = reinterpret_cast<weft_native_entry>(symbolAddress(handle, symbol));
    dispatch = reinterpret_cast<Dispatch>(symbolAddress(handle, dispatchSymbol));
  } catch (...) {
    dlclose(handle);
    throw;
  }
  library = handle;
}

Kernel::~Kernel() {
  if (library && dlclose(library) != 0) {
    std::fprintf(stderr, "Weft library release failed: %s\n", dlerror());
    std::terminate();
  }
}

} // namespace weft::runtime
