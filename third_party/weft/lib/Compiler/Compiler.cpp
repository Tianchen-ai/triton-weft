#include "weft/compiler.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelInterface.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCV/RISCVCompilation.h"
#include "Weft/Target/RISCV/RISCVCompiler.h"
#include "Weft/Target/RISCV/RISCVExtensions.h"
#include "Weft/Target/RISCV/RISCVTargetCatalog.h"
#include "Weft/Target/X86/X86Compiler.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>

using namespace mlir;
using namespace llvm;

struct weft_compiler_context {
  std::shared_ptr<MLIRContext> context;
};

struct weft_compiler_module {
  std::shared_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  weft_compiler_module(std::shared_ptr<MLIRContext> context, OwningOpRef<ModuleOp> module)
      : context(std::move(context)), module(std::move(module)) {}
};

namespace {
char *copyText(StringRef text) {
  auto result = std::make_unique<char[]>(text.size() + 1);
  std::memcpy(result.get(), text.data(), text.size());
  result[text.size()] = 0;
  return result.release();
}

std::string jsonText(const json::Value &value) {
  std::string result;
  raw_string_ostream(result) << value;
  return result;
}

template <typename F> int boundary(char **error, F action) {
  if (error) *error = nullptr;
  try {
    action();
    return 0;
  } catch (const std::exception &exception) {
    if (error) *error = copyText(exception.what());
    return 1;
  }
}

void require(bool condition, const char *message) {
  if (!condition) throw std::invalid_argument(message);
}

class Diagnostics {
  std::string text;
  ScopedDiagnosticHandler handler;
public:
  explicit Diagnostics(MLIRContext *context)
      : handler(context, [&](Diagnostic &diagnostic) {
          raw_string_ostream stream(text);
          diagnostic.print(stream);
          stream << '\n';
          return success();
        }) {}
  void check(bool valid, StringRef reason) {
    if (!valid) throw std::invalid_argument(reason.str() + (text.empty() ? "" : ":\n" + text));
  }
};

class CanonicalBuilder {
  MLIRContext &context;
  std::map<int64_t, mlir::Value> values;
  Type type(const char *text) {
    require(text, "Canonical type is absent");
    auto result = parseType(text, &context);
    require(bool(result), "invalid Canonical type");
    return result;
  }
  void bind(const weft_ir_value &description, mlir::Value value) {
    require(values.emplace(description.id, value).second, "duplicate Canonical SSA identity");
  }
  void populate(Region &region, const weft_ir_region &description, Location location) {
    require(!description.argument_count || description.arguments, "Canonical block arguments are absent");
    require(!description.operation_count || description.operations, "Canonical block operations are absent");
    auto *block = new Block;
    region.push_back(block);
    for (size_t i = 0; i < description.argument_count; ++i) {
      const auto &arg = description.arguments[i];
      bind(arg, block->addArgument(type(arg.type), location));
    }
    for (size_t i = 0; i < description.operation_count; ++i)
      block->push_back(operation(description.operations[i]));
  }
  Operation *operation(const weft_ir_operation &node) {
    require(node.name && node.filename, "Canonical operation name/location is absent");
    require(!node.operand_count || node.operands, "Canonical operands are absent");
    require(!node.result_count || node.results, "Canonical results are absent");
    require(!node.attribute_count || node.attributes, "Canonical attributes are absent");
    require(!node.region_count || node.regions, "Canonical regions are absent");
    auto location = FileLineColLoc::get(&context, node.filename, node.line, node.column);
    OperationState state(location, node.name);
    for (size_t i = 0; i < node.operand_count; ++i) {
      auto found = values.find(node.operands[i]);
      if (found == values.end())
        throw std::invalid_argument("undefined Canonical SSA operand " + std::to_string(node.operands[i]) + " of " + node.name);
      state.addOperands(found->second);
    }
    for (size_t i = 0; i < node.result_count; ++i) state.addTypes(type(node.results[i].type));
    for (size_t i = 0; i < node.attribute_count; ++i) {
      const auto &item = node.attributes[i];
      require(item.name && item.value, "Canonical attribute name/value is absent");
      auto attribute = parseAttribute(item.value, &context);
      require(bool(attribute), "invalid Canonical attribute");
      state.addAttribute(item.name, attribute);
    }
    for (size_t i = 0; i < node.region_count; ++i) populate(*state.addRegion(), node.regions[i], location);
    auto *op = Operation::create(state);
    for (size_t i = 0; i < node.result_count; ++i) bind(node.results[i], op->getResult(i));
    return op;
  }
public:
  explicit CanonicalBuilder(MLIRContext &context) : context(context) {}
  OwningOpRef<ModuleOp> build(StringRef name, const weft_ir_operation *ops, size_t count) {
    require(!count || ops, "Canonical module operations are absent");
    OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
    (*module)->setAttr("weft.source_module", StringAttr::get(&context, name));
    for (size_t i = 0; i < count; ++i) module->push_back(operation(ops[i]));
    return module;
  }
};

json::Object parseObject(const char *text) {
  require(text, "compiler options are absent");
  auto parsed = json::parse(text);
  if (!parsed) throw std::invalid_argument(toString(parsed.takeError()));
  auto *object = parsed->getAsObject();
  require(object, "compiler options must be an object");
  return std::move(*object);
}

StringRef text(const json::Object &object, StringRef name) {
  auto value = object.getString(name);
  if (!value) throw std::invalid_argument("missing string compiler option: " + name.str());
  return *value;
}
int64_t integer(const json::Object &object, StringRef name) {
  auto value = object.getInteger(name);
  if (!value) throw std::invalid_argument("missing integer compiler option: " + name.str());
  return *value;
}
StringMap<int64_t> meta(const json::Object &object) {
  auto *bindings = object.getObject("meta");
  require(bindings, "compiler source bindings are absent");
  StringMap<int64_t> result;
  for (const auto &item : *bindings) {
    auto value = item.second.getAsInteger();
    require(bool(value), "compiler source binding must be an integer");
    result[item.first] = *value;
  }
  return result;
}

weft::RISCVCompilerOptions riscvOptions(const json::Object &values) {
  weft::RISCVCompilerOptions result;
  std::string error;
  if (!weft::parseRISCVTargetProfile(text(values, "march"), text(values, "abi"), integer(values, "vlen_bits"),
      text(values, "matrix_extension"), integer(values, "max_widening_combine_groups"),
      text(values, "partial_combine_policy"), text(values, "record_axis_policy"), result.target, error))
    throw std::invalid_argument(error);
  result.target.maxPrivateStackBytes = integer(values, "private_stack_bytes");
  auto *cacheValue = values.get("data_cache");
  require(cacheValue, "data_cache option is absent");
  if (auto *cache = cacheValue->getAsObject()) {
    result.target.dataCache = weft::RISCVDataCache{integer(*cache,"size_bytes"), integer(*cache,"line_bytes"),
                                                integer(*cache,"ways"), integer(*cache,"sets")};
    if (!result.target.dataCache->validate(error)) throw std::invalid_argument(error);
  } else require(cacheValue->kind() == json::Value::Null, "data_cache must be an object or null");
  auto *required = values.getArray("required_extensions");
  require(required, "required_extensions option is absent");
  for (const auto &name : *required) {
    auto value = name.getAsString();
    require(bool(value), "required extension must be a string");
    result.requiredExtensions.push_back(value->str());
  }
  result.localTileStrategy = text(values,"local_tile_strategy").str();
  result.lmulEighths = integer(values,"lmul_eighths");
  result.unroll = integer(values,"unroll");
  result.pipelineDepth = integer(values,"pipeline_depth");
  result.nativeLoopAlignment = integer(values,"native_loop_alignment");
  result.scalarLoadPrime = integer(values,"scalar_load_prime");
  result.pointwiseIssueCopies = integer(values,"pointwise_issue_copies");
  result.contractionColumnCopies = integer(values,"contraction_column_copies");
  result.panelTransferFields = integer(values,"panel_transfer_fields");
  result.panelTransferStrategy = text(values,"panel_transfer_strategy").str();
  result.contractionPanelK = integer(values,"contraction_panel_k");
  result.contractionKSteps = integer(values,"contraction_k_steps");
  result.metaBindings = meta(values);
  if (!result.validateBindings(error)) throw std::invalid_argument(error);
  return result;
}

json::Value attributeJSON(Attribute attribute) {
  if (auto v = dyn_cast<BoolAttr>(attribute)) return v.getValue();
  if (auto v = dyn_cast<IntegerAttr>(attribute)) return v.getInt();
  if (auto v = dyn_cast<FloatAttr>(attribute)) return v.getValueAsDouble();
  if (auto v = dyn_cast<StringAttr>(attribute)) return v.getValue().str();
  if (auto a = dyn_cast<ArrayAttr>(attribute)) {
    json::Array values;
    for (auto v : a) values.push_back(attributeJSON(v));
    return values;
  }
  if (auto d = dyn_cast<DictionaryAttr>(attribute)) {
    json::Object values;
    for (auto v : d) values[v.getName().str()] = attributeJSON(v.getValue());
    return values;
  }
  throw std::invalid_argument("unsupported physical schedule attribute");
}

template <typename T> json::Array array(const T &values) {
  json::Array result;
  for (const auto &value : values) result.push_back(value);
  return result;
}

json::Object layoutJSON(const weft::RISCVLayoutDiagnostic &l) {
  return json::Object{{"role",l.role},{"index",l.index},{"element_type",l.elementType},{"carrier",l.carrier},
    {"validity",l.validity},{"shape",array(l.shape)},{"axis_ids",array(l.axes)},
    {"layout_axis_ids",array(l.layoutAxes)},{"time_factors",array(l.timeFactors)},
    {"lane_factors",array(l.laneFactors)},{"replica_factors",array(l.replicaFactors)},
    {"fragment_factors",array(l.fragmentFactors)},{"local_factors",array(l.localFactors)},
    {"sew",l.sew},{"lmul_eighths",l.lmulEighths},{"vl",l.vl},
    {"register_groups",l.registerGroups},{"replica_group",l.replicaGroup}};
}

json::Value accessJSON(const std::optional<weft::RISCVAccessDiagnostic> &a) {
  if (!a) return nullptr;
  return json::Object{{"form",a->form},{"mapping",a->mapping},{"alignment",a->alignment},
    {"index_sew",a->indexSEW},{"segment_fields",a->segmentFields},{"group_size",a->groupSize},
    {"layer_size",a->layerSize},{"join_fields",a->joinFields},{"join_low_bits",a->joinLowBits},
    {"join_role",a->joinRole},{"bit_offset",a->bitOffset},{"storage_bits",a->storageBits},
    {"order",a->order},{"field_chunk_bytes",a->fieldChunkBytes}};
}

json::Object physicalJSON(const weft::RISCVKernelABI &kernel) {
  json::Array operations;
  int64_t count = 0;
  for (const auto &op : kernel.physicalOperations) {
    json::Array layouts;
    for (const auto &layout : op.layouts) layouts.push_back(layoutJSON(layout));
    count += op.occurrences;
    json::Value packet(nullptr);
    if (op.scalarPacket) {
      const auto &p = *op.scalarPacket;
      packet = json::Object{{"axis",p.axis},{"steps",p.steps},{"parts",p.parts},
                           {"registers",p.registers},{"live_registers",p.liveRegisters}};
    }
    operations.push_back(json::Object{{"operation",op.operation},{"engine",op.engine},
      {"family",op.family},{"instruction",op.instruction},{"mask",op.mask},{"tail",op.tail},
      {"access",accessJSON(op.access)},{"scalar_packet",std::move(packet)},{"occurrences",op.occurrences},
      {"operand_count",op.operandCount},{"result_count",op.resultCount},{"operand_groups",op.operandGroups},
      {"result_groups",op.resultGroups},{"temporary_groups",op.temporaryGroups},
      {"fragment_groups",op.fragmentGroups},{"local_bytes",op.localBytes},
      {"parameters",array(op.parameters)},{"locations",array(op.locations)},{"layouts",std::move(layouts)}});
  }
  return json::Object{{"scope","final-physical-ir/static-selected-operations"},{"selected_operation_count",count},
          {"operations",std::move(operations)}};
}

json::Object kernelJSON(const weft::RISCVKernelABI &kernel) {
  json::Object bindings, realizations, isa, resources;
  for (const auto &[name,value] : kernel.bindings) bindings[name] = value;
  for (const auto &[name,value] : kernel.localTileRealizations) realizations[name] = value;
  for (const auto &[name,value] : kernel.isaExtensions)
    isa[name] = json::Array{value.first,value.second};
  for (const auto &[name,value] : kernel.physicalResources) resources[name] = value;
  json::Value cache(nullptr);
  if (kernel.dataCache) {
    const auto &c = *kernel.dataCache;
    cache = json::Object{{"size_bytes",c.sizeBytes},{"line_bytes",c.lineBytes},{"ways",c.ways},{"sets",c.sets}};
  }
  json::Array arguments;
  for (const auto &a : kernel.arguments)
    arguments.push_back(json::Object{{"name",a.name},{"kind",a.kind},{"dtype",a.dtype},{"access",a.access},
      {"strided",a.strided},{"c_type",a.cType},{"encoding",a.encoding},{"shape",array(a.shape)},
      {"storage_bytes",a.storageBytes},{"record_elements",a.recordElements},{"interleave_rows",a.interleaveRows},
      {"alignment",a.alignment},{"alias_set",a.aliasSet},{"writable",a.writable}});
  return json::Object{{"symbol",kernel.symbol},{"declaration",kernel.declaration},{"march",kernel.march},
    {"abi",kernel.abi},{"vlen_bits",kernel.vlenBits},{"isa_extensions",std::move(isa)},
    {"matrix_extensions",array(kernel.matrixExtensions)},{"required_extensions",array(kernel.requiredExtensions)},
    {"used_extensions",array(kernel.usedExtensions)},{"shape_parameters",array(kernel.shapeParameters)},
    {"bindings",std::move(bindings)},{"physical_resources",std::move(resources)},
    {"physical_operations",physicalJSON(kernel)},{"local_tile_strategy",kernel.localTileStrategy},
    {"local_tile_realizations",std::move(realizations)},{"data_cache",std::move(cache)},
    {"arguments",std::move(arguments)}};
}

void reportFailure(const weft::RISCVCompilationReport &report) {
  throw std::runtime_error(jsonText(report.failureJSON()));
}

json::Object lowerX86(ModuleOp source, const json::Object &options, OwningOpRef<ModuleOp> &result) {
  weft::X86Target target;
  std::string error;
  if (!weft::parseX86Target(text(options,"march").str(),text(options,"abi").str(),integer(options,"vlen_bits"),target,error))
    throw std::invalid_argument(error);
  auto tune = text(options,"tune_cpu");
  auto depth = integer(options,"pipeline_depth");
  require(depth >= 1 && depth <= 4, "pipeline_depth must be one of 1, 2, 3, 4");
  weft::RISCVCompilationReport report;
  report.begin("canonical-to-x86");
  report.startPass("source-specialization-and-physical-lowering","weft-bind-source-parameters","canonical-to-x86");
  ScopedDiagnosticHandler handler(source.getContext(), [&](Diagnostic &diag) { report.capture(diag); return success(); });
  auto lowered = weft::lowerX86Module(source,target,meta(options),tune,depth);
  report.finishPass(succeeded(lowered));
  if (failed(lowered)) { report.captureFailedIR(source); reportFailure(report); }
  result = std::move(*lowered);
  auto object = report.compilationJSON();
  object["pipelines"] = attributeJSON((*result)->getAttr("weft.x86.pipelines"));
  auto facts = weft::x86TargetJSON(target);
  object["target_resources"] = std::move(*facts.getObject("register_file"));
  object["tuning_cpu"] = tune.empty() ? target.baseCPU() : tune.str();
  object["resource_estimate_scope"] = "physical array candidates with enclosing SSA residency; final register allocation and spills belong to LLVM";
  json::Array contractions, schedules, packing, storage;
  if (auto entries = (*result)->getAttrOfType<ArrayAttr>("weft.x86.contractions"))
    for (auto entry : entries) contractions.push_back(attributeJSON(entry));
  int64_t forwarded = 0, uniform = 0;
  for (auto function : result->getOps<func::FuncOp>()) {
    if (auto count = function->getAttrOfType<IntegerAttr>("weft.x86.promoted_arrays"))
      storage.push_back(json::Object{{"function",function.getSymName().str()},{"promoted_arrays",count.getInt()},
        {"reserved_bytes",function->getAttrOfType<IntegerAttr>("weft.x86.stack_storage_bytes").getInt()},
        {"budget_bytes",function->getAttrOfType<IntegerAttr>("weft.x86.stack_storage_budget").getInt()}});
    if (auto count = function->getAttrOfType<IntegerAttr>("weft.x86.uniform_validity_arrays")) uniform += count.getInt();
    if (auto count = function->getAttrOfType<IntegerAttr>("weft.x86.forwarded_arrays")) forwarded += count.getInt();
    for (auto [name, destination] : {std::pair{"weft.x86.array_schedule",&schedules},
                                     std::pair{"weft.x86.packing_schedule",&packing}})
      if (auto entries = function->getAttrOfType<ArrayAttr>(name))
        for (auto entry : entries) {
          auto value = attributeJSON(entry);
          (*value.getAsObject())["function"] = function.getSymName().str();
          destination->push_back(std::move(value));
        }
  }
  object["contractions"] = std::move(contractions);
  object["array_schedules"] = std::move(schedules);
  object["packing_schedules"] = std::move(packing);
  object["local_storage"] = std::move(storage);
  object["forwarded_arrays"] = forwarded;
  object["uniform_validity_arrays"] = uniform;
  object["array_ownership_lowered"] = (*result)->hasAttr("weft.x86.dynamic_storage");
  return object;
}
} // namespace

extern "C" {
unsigned weft_compiler_abi(void) { return 1; }
void weft_compiler_string_release(char *text) { delete[] text; }

int weft_compiler_context_create(weft_compiler_context **result, char **error) {
  return boundary(error, [&] {
    require(result, "compiler context output is absent");
    auto context = std::make_shared<MLIRContext>();
    weft::loadX86Dialects(*context);
    context->loadDialect<weft::kernel::WEFTKernelDialect,weft::riscv::WEFTRISCVDialect,
                         arith::ArithDialect,scf::SCFDialect,func::FuncDialect>();
    *result = new weft_compiler_context{std::move(context)};
  });
}
void weft_compiler_context_release(weft_compiler_context *context) { delete context; }

int weft_compiler_module_build(weft_compiler_context *context, const char *name,
    const weft_ir_operation *ops, size_t count, weft_compiler_module **result, char **error) {
  return boundary(error, [&] {
    require(context && name && result, "compiler module inputs are absent");
    Diagnostics diagnostics(context->context.get());
    auto module = CanonicalBuilder(*context->context).build(name,ops,count);
    diagnostics.check(succeeded(verify(*module)),"invalid Weft Canonical program");
    *result = new weft_compiler_module(context->context,std::move(module));
  });
}
int weft_compiler_module_parse(weft_compiler_context *context, const char *text,
    weft_compiler_module **result, char **error) {
  return boundary(error, [&] {
    require(context && text && result, "compiler IR input is absent");
    Diagnostics diagnostics(context->context.get());
    auto module = parseSourceString<ModuleOp>(text,context->context.get());
    diagnostics.check(bool(module),"invalid Weft IR");
    *result = new weft_compiler_module(context->context,std::move(module));
  });
}
void weft_compiler_module_release(weft_compiler_module *module) { delete module; }
int weft_compiler_module_verify(weft_compiler_module *module, char **error) {
  return boundary(error,[&] {
    require(module,"compiler module is absent");
    Diagnostics diagnostics(module->context.get());
    diagnostics.check(succeeded(verify(*module->module)),"invalid Weft program");
  });
}
int weft_compiler_module_text(weft_compiler_module *module, char **result, char **error) {
  return boundary(error,[&] {
    require(module && result,"compiler module/output is absent");
    std::string text;
    raw_string_ostream stream(text);
    module->module->print(stream);
    *result = copyText(text);
  });
}
int weft_compiler_module_location_snapshot(weft_compiler_module *module, const char *filename, char **error) {
  return boundary(error,[&] {
    require(module && filename,"compiler module/snapshot filename is absent");
    Diagnostics diagnostics(module->context.get());
    diagnostics.check(succeeded(generateLocationsFromIR(filename, *module->module, OpPrintingFlags())),
                      "failed to create Weft IR location snapshot");
  });
}
int weft_compiler_module_interface(weft_compiler_module *module, char **result, char **error) {
  return boundary(error,[&] {
    require(module && result,"compiler module/output is absent");
    Diagnostics diagnostics(module->context.get());
    auto description = weft::kernel::describeKernelInterface(*module->module);
    diagnostics.check(succeeded(description),"invalid Canonical kernel interface");
    *result = copyText(*description);
  });
}
int weft_compiler_module_lower(weft_compiler_module *source, const char *optionsText,
    weft_compiler_module **result, char **metadata, char **error) {
  return boundary(error,[&] {
    require(source && result && metadata,"compiler lower input/output is absent");
    auto options = parseObject(optionsText);
    auto architecture = text(options,"arch");
    OwningOpRef<ModuleOp> module;
    json::Object report;
    if (architecture == "x86_64") report = lowerX86(*source->module,options,module);
    else if (architecture == "riscv64") {
      weft::RISCVCompilationReport diagnostics;
      auto lowered = weft::lowerRISCVModule(*source->module,riscvOptions(options),&diagnostics);
      if (failed(lowered)) reportFailure(diagnostics);
      module = std::move(*lowered);
      report = diagnostics.compilationJSON();
      json::Array supplies;
      module->walk([&](weft::riscv::KernelOp kernel) {
        if (auto choices = kernel->getAttrOfType<ArrayAttr>("weft.riscv.residency_choices"))
          for (auto choice : choices) {
            auto value = attributeJSON(choice);
            (*value.getAsObject())["kernel"] = kernel.getSymName().str();
            supplies.push_back(std::move(value));
          }
      });
      report["supply_choices"] = std::move(supplies);
    } else throw std::invalid_argument("unsupported Weft architecture: " + architecture.str());
    *metadata = copyText(jsonText(std::move(report)));
    *result = new weft_compiler_module(source->context,std::move(module));
  });
}
int weft_compiler_module_emit(weft_compiler_module *module, char **result, char **error) {
  return boundary(error,[&] {
    require(module && result,"compiler emit input/output is absent");
    weft::RISCVCompilationReport report;
    json::Object output;
    if ((*module->module)->hasAttr("weft.x86.abi")) {
      report.begin("x86-to-llvm");
      report.startPass("translation","x86-to-llvm","arith-math-vector-func-to-llvm");
      ScopedDiagnosticHandler handler(module->context.get(),[&](Diagnostic &d) { report.capture(d); return success(); });
      auto emitted = weft::translateX86Module(*module->module);
      report.finishPass(succeeded(emitted));
      if (failed(emitted)) { report.captureFailedIR(*module->module); reportFailure(report); }
      output["llvm_ir"] = *emitted;
      auto abi = json::parse((*module->module)->getAttrOfType<StringAttr>("weft.x86.abi").getValue());
      if (!abi) throw std::runtime_error(toString(abi.takeError()));
      output["kernel"] = std::move(*abi);
    } else {
      auto emitted = weft::translateRISCVModule(*module->module,&report);
      if (failed(emitted)) reportFailure(report);
      output["intrinsic_c"] = emitted->intrinsicC;
      json::Array kernels;
      for (const auto &kernel : emitted->kernels) kernels.push_back(kernelJSON(kernel));
      output["kernels"] = std::move(kernels);
    }
    output["compilation"] = report.compilationJSON();
    *result = copyText(jsonText(std::move(output)));
  });
}
int weft_compiler_target_info(const char *query, const char *optionsText, char **result, char **error) {
  return boundary(error,[&] {
    require(query && result,"compiler target query/output is absent");
    auto options = parseObject(optionsText);
    json::Value output(nullptr);
    std::string failure;
    if (StringRef(query) == "extension_catalog") {
      if (!weft::validateRISCVExtensionRegistry(failure)) throw std::invalid_argument(failure);
      output = weft::riscvExtensionCatalogJSON();
    } else if (StringRef(query) == "x86_target_capabilities") {
      weft::X86Target target;
      if (!weft::parseX86Target(text(options,"march").str(),text(options,"abi").str(),integer(options,"vlen_bits"),target,failure))
        throw std::invalid_argument(failure);
      output = weft::x86TargetJSON(target);
    } else if (StringRef(query) == "target_capabilities") {
      auto configuration = riscvOptions(options);
      json::Object facts;
      if (!weft::riscvTargetCapabilitiesJSON(configuration.target,facts,failure)) throw std::invalid_argument(failure);
      output = std::move(facts);
    } else if (StringRef(query) == "native_target") {
#if defined(__x86_64__)
      weft::X86NativeTarget native;
      if (!weft::queryNativeX86Target(native,failure)) throw std::invalid_argument(failure);
      auto facts = weft::x86TargetJSON(native.target);
      facts["cpus"] = array(native.cpus);
      facts["available_isa_extensions"] = array(native.availableFeatures);
      json::Object matrix;
      for (auto name : {"amx-bf16","amx-int8"})
        if (llvm::is_contained(native.availableFeatures,name)) matrix[name] = array(native.cpus);
      facts["native_matrix_extensions"] = std::move(matrix);
      output = std::move(facts);
#else
      weft::RISCVNativeTarget native;
      if (!weft::queryNativeRISCVTarget(native,failure)) throw std::invalid_argument(failure);
      json::Object facts;
      if (!weft::riscvTargetCapabilitiesJSON(native.profile,facts,failure)) throw std::invalid_argument(failure);
      facts["cpus"] = array(native.cpus);
      json::Object matrix;
      for (const auto &[name,cpus] : native.matrixExtensions) matrix[name] = array(cpus);
      facts["native_matrix_extensions"] = std::move(matrix);
      json::Object extensions;
      for (const auto &[name, version] : native.isaExtensions)
        extensions[name] = json::Array{version.first, version.second};
      facts["execution_isa_extensions"] = std::move(extensions);
      output = std::move(facts);
#endif
    } else throw std::invalid_argument("unknown Weft target query");
    *result = copyText(jsonText(output));
  });
}
} // extern C
