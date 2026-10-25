#include "Weft/Target/X86Compiler.h"
#include "Weft/Target/SourceSpecialization.h"
#include "Weft/Analysis/LogicalAxes.h"
#include "Weft/Analysis/ReductionIdentity.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelInterface.h"
#include "Weft/Analysis/StorageBytes.h"
#include "Arrays.h"
#include "ArrayPackets.h"
#include "IntegerReductions.h"
#include "Lookups.h"
#include "AMXContractions.h"
#include "ContractionEngines.h"
#include "Math.h"
#include "IndexRelations.h"
#include <numeric>
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/X86TargetParser.h"
#include <algorithm>
#include <numeric>
#include <limits>

using namespace mlir;
namespace wk = weft::kernel;
namespace {

InFlightDiagnostic unsupported(Operation *op, const Twine &message) {
  auto diagnostic = op->emitError(message);
  diagnostic.getUnderlyingDiagnostic()->getMetadata().emplace_back(
      StringRef("weft.candidate.unsupported"));
  return diagnostic;
}
InFlightDiagnostic unsupported(Operation &op, const Twine &message) {
  return unsupported(&op, message);
}

// The SDK's masked LLVM transfers do not expose MemoryEffectOpInterface. Their
// read/write effects must stay visible when array ownership is analyzed beside
// fixed SIMD operations; neither operation allocates or releases a local buffer.
template <typename Op, bool Read>
struct TransferEffects : MemoryEffectOpInterface::ExternalModel<TransferEffects<Op, Read>, Op> {
  void getEffects(Operation *, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    if constexpr (Read) effects.emplace_back(MemoryEffects::Read::get());
    else effects.emplace_back(MemoryEffects::Write::get());
  }
};

Type element(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return value.getElementType();
  if (auto vector = dyn_cast<VectorType>(type)) return vector.getElementType();
  return type;
}
Type scalar(Type type) {
  type = element(type);
  if (auto integer = dyn_cast<IntegerType>(type))
    return IntegerType::get(type.getContext(), integer.getWidth());
  return type;
}
bool unsignedType(Type type) {
  auto integer = dyn_cast<IntegerType>(element(type));
  return integer && integer.isUnsigned();
}
SmallVector<int64_t> logicalShape(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type))
    return llvm::to_vector(value.getShape().asArrayRef());
  if (auto view = dyn_cast<wk::ViewType>(type)) return llvm::to_vector(view.getShape().asArrayRef());
  if (auto view = dyn_cast<wk::SliceType>(type)) return llvm::to_vector(view.getShape().asArrayRef());
  return {};
}
SmallVector<int64_t> axes(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type))
    return llvm::to_vector(value.getAxisIds().asArrayRef());
  if (auto view = dyn_cast<wk::ViewType>(type)) return llvm::to_vector(view.getAxisIds().asArrayRef());
  if (auto view = dyn_cast<wk::SliceType>(type)) return llvm::to_vector(view.getAxisIds().asArrayRef());
  return {};
}
int64_t volume(ArrayRef<int64_t> dimensions) {
  return std::accumulate(dimensions.begin(), dimensions.end(), int64_t{1}, std::multiplies<int64_t>());
}
bool memoryType(Type type) { return isa<wk::ViewType, wk::SliceType>(type); }
wk::EncodingType packedType(Type type) {
  if (auto view = dyn_cast<wk::ViewType>(type)) type = view.getEncoding();
  if (auto view = dyn_cast<wk::SliceType>(type)) type = view.getEncoding();
  auto encoding = dyn_cast<wk::EncodingType>(element(type));
  return encoding && encoding.getKind() != "dense" ? encoding : wk::EncodingType();
}
Type denseElement(Type type) {
  Type encoding;
  if (auto view = dyn_cast<wk::ViewType>(type)) encoding = view.getEncoding();
  if (auto view = dyn_cast<wk::SliceType>(type)) encoding = view.getEncoding();
  auto dense = dyn_cast_or_null<wk::EncodingType>(encoding);
  if (!dense || dense.getKind() != "dense") return {};
  auto *context = type.getContext();
  StringRef family = dense.getFamily();
  if (family == "f16") return Float16Type::get(context);
  if (family == "bf16") return BFloat16Type::get(context);
  if (family == "f32") return Float32Type::get(context);
  if (family == "f64") return Float64Type::get(context);
  unsigned bits;
  if (family.size() < 2 || (family.front() != 'i' && family.front() != 'u') ||
      family.drop_front().getAsInteger(10, bits) ||
      !llvm::is_contained(ArrayRef<unsigned>{1, 8, 16, 32, 64}, bits)) return {};
  return IntegerType::get(context, bits, bits == 1 ? IntegerType::Signless :
      family.front() == 'u' ? IntegerType::Unsigned : IntegerType::Signed);
}

// The physical module uses fixed vectors and explicit LLVM memory operations.
// Axis relations are consumed here, before LLVM instruction selection. LLVM
// does not have to recover the author's broadcasting or memory mask contract.
class Lowering {
public:
  Lowering(ModuleOp module, const weft::X86Target &target)
      : module(module), target(target), b(module.getContext()) {}

  LogicalResult run(wk::KernelOp kernel) {
    sourceModule = kernel->getParentOfType<ModuleOp>();
    shapeSymbols = kernel.getShapeSymbols();
    auto description = wk::describeKernelInterface(kernel->getParentOfType<ModuleOp>());
    if (failed(description)) return failure();
    auto parsed = llvm::json::parse(*description);
    if (!parsed) return kernel.emitError("cannot read verified canonical interface");
    auto &interface = *parsed->getAsObject()->getObject("interface");
    for (auto &entry : *interface.getObject("source_parameters")) {
      auto &choices = *entry.second.getAsArray();
      if (choices.size() != 1)
        return kernel.emitError("source parameter must be bound before x86 physical lowering: ") << StringRef(entry.first);
      sourceParameters[entry.first] = *choices[0].getAsInteger();
    }
    for (auto [index, attribute] : llvm::enumerate(kernel.getShapeSymbols())) {
      auto selected = sourceParameters.find(cast<StringAttr>(attribute).getValue());
      if (selected != sourceParameters.end())
        shapeBindings[-static_cast<int64_t>(index) - 1] = selected->second;
    }
    auto bounded = kernel.walk([&](Operation *op) {
      for (Type type : op->getResultTypes())
        if (isa<wk::ValueType>(type) && !physicalType(type)) {
          unsupported(op, "x86 local representation requires numeric elements or a closed encoded record layout");
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    });
    if (bounded.wasInterrupted()) return failure();
    SmallVector<Type> parameters;
    for (auto arg : kernel.getBody().front().getArguments()) {
      Type type = memoryType(arg.getType()) ? Type(LLVM::LLVMPointerType::get(b.getContext())) : physicalType(arg.getType());
      if (!type || isa<VectorType>(type))
        return unsupported(kernel, "unsupported x86 entry type: ") << arg.getType();
      parameters.push_back(type);
    }
    auto &shapeParameters = *interface.getArray("shape_parameters");
    for (auto &name : shapeParameters) parameters.push_back(b.getIndexType());
    for (auto &record : *interface.getArray("arguments"))
      if (*record.getAsObject()->getBoolean("strided"))
        for (auto &dimension : *record.getAsObject()->getArray("shape")) parameters.push_back(b.getIndexType());
    b.setInsertionPointToEnd(module.getBody());
    auto function = func::FuncOp::create(b, kernel.getLoc(), kernel.getSymName(), b.getFunctionType(parameters, {}));
    function->setAttr("weft.x86.cpu", b.getStringAttr(target.cpu));
    function->setAttr("weft.x86.vector_bits", b.getI32IntegerAttr(target.vectorBits));
    function->setAttr("weft.x86.vector_registers", b.getI32IntegerAttr(target.vectorRegisters()));
    // The native View contract checks disjoint alias groups before entry. Carry
    // singleton groups into LLVM rather than discarding that author contract.
    // Raw pointers have no checked View range, so a mixed ABI grants no noalias.
    bool checkedViews = true;
    DenseMap<int64_t, unsigned> aliasCounts;
    for (auto [index, arg] : llvm::enumerate(kernel.getBody().front().getArguments())) {
      if (isa<LLVM::LLVMPointerType>(parameters[index]) && !memoryType(arg.getType())) checkedViews = false;
      if (memoryType(arg.getType())) ++aliasCounts[kernel.getArgAliasSets()[index]];
    }
    kernel.walk([&](Operation *op) {
      if (isa<wk::PointerLoadOp, wk::PointerStoreOp, wk::PointerAtomicOp>(op)) checkedViews = false;
    });
    for (auto [index, arg] : llvm::enumerate(kernel.getBody().front().getArguments())) {
      if (!memoryType(arg.getType())) continue;
      if (checkedViews && aliasCounts[kernel.getArgAliasSets()[index]] == 1)
        function.setArgAttr(index, LLVM::LLVMDialect::getNoAliasAttrName(), b.getUnitAttr());
      if (cast<StringAttr>(kernel.getArgAccess()[index]).getValue() == "read")
        function.setArgAttr(index, LLVM::LLVMDialect::getReadonlyAttrName(), b.getUnitAttr());
      if (!packedType(arg.getType()) && cast<StringAttr>(kernel.getArgAccess()[index]).getValue() == "write")
        function.setArgAttr(index, "weft.x86.write_only", b.getUnitAttr());
    }
    Block *entry = function.addEntryBlock();
    for (auto [old, value] : llvm::zip(kernel.getBody().front().getArguments(), entry->getArguments()))
      values.map(old, value);
    b.setInsertionPointToEnd(entry);
    size_t extra = kernel.getBody().front().getNumArguments();
    for (auto &name : shapeParameters) dimensions[name.getAsString()->str()] = entry->getArgument(extra++);
    for (auto [index, old] : llvm::enumerate(kernel.getBody().front().getArguments())) {
      if (!memoryType(old.getType())) continue;
      auto &record = *(*interface.getArray("arguments"))[index].getAsObject();
      Memory view;
      view.base = entry->getArgument(index);
      view.offset = arith::ConstantIndexOp::create(b, kernel.getLoc(), 0);
      view.guard = constant(kernel.getLoc(), b.getI1Type(), b.getBoolAttr(true));
      for (int64_t extent : shape(old.getType())) {
        Value size = extent >= 0 ? Value(arith::ConstantIndexOp::create(b, kernel.getLoc(), extent)) :
            dimensions.lookup(cast<StringAttr>(kernel.getShapeSymbols()[-extent - 1]).getValue());
        if (!size) return kernel.emitError("x86 View has no entry shape binding");
        view.extents.push_back(size);
        view.origins.push_back(view.offset);
      }
      view.limits = view.extents;
      initializeStorage(view, kernel.getLoc());
      if (auto encoding = packedType(old.getType())) {
        auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
        if (failed(layout)) return unsupported(kernel, "x86 packed View has no closed storage layout");
        view.recordBytes = indexConstant(kernel.getLoc(), layout->bytes());
        view.recordColumns = arith::DivUIOp::create(b, kernel.getLoc(), view.extents.back(), indexConstant(kernel.getLoc(), layout->elements()));
        view.interleaveRows = indexConstant(kernel.getLoc(), layout->rows);
      }
      view.strides.resize(view.extents.size());
      if (*record.getBoolean("strided")) {
        for (auto &stride : view.strides) stride = entry->getArgument(extra++);
      } else {
        Value running = arith::ConstantIndexOp::create(b, kernel.getLoc(), 1);
        for (int64_t axis = view.strides.size() - 1; axis >= 0; --axis) {
          view.strides[axis] = running;
          if (axis > 0) running = b.createOrFold<arith::MulIOp>(kernel.getLoc(), running, view.extents[axis]);
        }
      }
      memories[old] = std::move(view);
    }
    if (failed(block(kernel.getBody().front()))) return failure();
    llvm::json::Array arguments;
    SmallVector<std::string> declarations;
    for (auto [index, old] : llvm::enumerate(kernel.getBody().front().getArguments())) {
      auto record = std::move(*(*interface.getArray("arguments"))[index].getAsObject());
      bool pointer = isa<wk::PointerType>(old.getType()) || memoryType(old.getType());
      Type type = packedType(old.getType()) ? Type(b.getIntegerType(8, IntegerType::Unsigned)) : memoryType(old.getType()) ? denseElement(old.getType()) :
          pointer ? cast<wk::PointerType>(old.getType()).getElementType() : old.getType();
      std::string ctype;
      if (type.isIndex()) ctype = "size_t";
      else if (type.isF32()) ctype = "float";
      else if (type.isF64()) ctype = "double";
      else if (type.isF16() && pointer) ctype = "_Float16";
      else if (type.isBF16() && pointer) ctype = "uint16_t";
      else if (auto integer = dyn_cast<IntegerType>(type))
        ctype = integer.getWidth() == 1 ? "bool" :
            (integer.isUnsigned() ? "uint" : "int") + std::to_string(integer.getWidth()) + "_t";
      else return unsupported(kernel, "unsupported x86 C ABI element type: ") << type;
      auto access = *record.getString("access");
      bool writable = access == "write" || access == "readwrite";
      if (pointer) ctype = (writable ? "" : "const ") + ctype + " *";
      record["c_type"] = ctype;
      record["writable"] = writable;
      int64_t bytes = type.isIndex() ? 8 : std::max<unsigned>(8, type.getIntOrFloatBitWidth()) / 8;
      record["storage_bytes"] = pointer ? bytes : 0;
      record["record_elements"] = pointer ? 1 : 0;
      record["interleave_rows"] = 0;
      record["alignment"] = pointer ? bytes : 0;
      if (auto encoding = packedType(old.getType())) {
        auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
        if (failed(layout)) return unsupported(kernel, "x86 packed ABI has no storage declaration");
        record["storage_bytes"] = layout->bytes();
        record["record_elements"] = layout->elements();
        record["interleave_rows"] = layout->rows > 1 ? layout->rows : 0;
        record["alignment"] = layout->base.getAlignment();
        if (layout->rows > 1) {
          std::string prefix = kernel.getSymName().str() + "_" + record.getString("name")->str();
          record["artifact_pack"] = llvm::json::Object{
              {"base_encoding", layout->base.getSymName().str()},
              {"size_symbol", prefix + "_packed_size"}, {"pack_symbol", prefix + "_pack"}};
        }
      }
      declarations.push_back(ctype + " " + record.getString("name")->str());
      arguments.push_back(std::move(record));
    }
    // The shared SDK supplies all shape scalars, then declared per-View strides.
    // Keep extra names independent of user identifiers and of source symbols.
    for (size_t index = arguments.size(); index < parameters.size(); ++index) {
      std::string name = "__weft_dimension_" + std::to_string(index);
      while (llvm::any_of(arguments, [&](auto &arg) { return *arg.getAsObject()->getString("name") == name; })) name += "_";
      declarations.push_back("size_t " + name);
    }
    llvm::json::Object isa;
    for (const auto &feature : target.features) isa[feature] = llvm::json::Array{1, 0};
    llvm::json::Object selectedParameters;
    for (auto &entry : sourceParameters) selectedParameters[entry.getKey()] = entry.second;
    llvm::json::Array matrixExtensions, usedMatrixExtensions;
    for (auto feature : {"amx-bf16", "amx-int8"})
      if (target.hasFeature(feature)) matrixExtensions.push_back(feature);
    if (module.lookupSymbol<func::FuncOp>("llvm.x86.tdpbf16ps")) usedMatrixExtensions.push_back("amx-bf16");
    for (auto name : {"llvm.x86.tdpbssd", "llvm.x86.tdpbsud", "llvm.x86.tdpbusd", "llvm.x86.tdpbuud"})
      if (module.lookupSymbol<func::FuncOp>(name)) { usedMatrixExtensions.push_back("amx-int8"); break; }
    llvm::json::Object abi{{"symbol", kernel.getSymName()},
        {"declaration", "void " + kernel.getSymName().str() + "(" + llvm::join(declarations, ", ") + ");"},
        {"arch", "x86_64"}, {"march", target.cpu}, {"abi", "sysv"}, {"vlen_bits", 0},
        {"vector_bits", target.vectorBits}, {"isa_extensions", std::move(isa)},
        {"matrix_extensions", std::move(matrixExtensions)},
        {"used_extensions", std::move(usedMatrixExtensions)},
        {"required_extensions", llvm::json::Array{}}, {"shape_parameters", std::move(shapeParameters)},
        {"bindings", std::move(selectedParameters)}, {"arguments", std::move(arguments)}};
    module->setAttr("weft.x86.abi", b.getStringAttr(llvm::formatv("{0}", llvm::json::Value(std::move(abi))).str()));
    for (auto [index, argument] : llvm::enumerate(kernel.getBody().front().getArguments()))
      if (auto encoding = packedType(argument.getType()); encoding && encoding.getKind() == "derived_instance")
        if (failed(artifactBuilder(kernel, index, encoding))) return failure();
    return success();
  }

private:
  ModuleOp module;
  ModuleOp sourceModule;
  const weft::X86Target &target;
  OpBuilder b;
  IRMapping values;
  func::FuncOp cfgFunction;
  llvm::StringMap<int64_t> sourceParameters;
  DenseMap<int64_t, int64_t> shapeBindings;
  DenseMap<int64_t, Value> localExtents;
  DenseMap<Value, SmallVector<Value>> arrayShapes;
  DenseMap<Block *, int64_t> arrayScopes;
  IntegerAttr arrayScope(Operation &op) {
    auto [entry, inserted] = arrayScopes.try_emplace(op.getBlock(), arrayScopes.size());
    return b.getI64IntegerAttr(entry->second);
  }
  ArrayAttr shapeSymbols;
  SmallVector<int64_t> shape(Type type) const {
    auto result = logicalShape(type);
    for (int64_t &extent : result) {
      auto selected = shapeBindings.find(extent);
      if (selected != shapeBindings.end()) extent = selected->second;
    }
    return result;
  }
  Type physicalType(Type type) const {
    if (isa<wk::PointerType>(type)) return LLVM::LLVMPointerType::get(type.getContext());
    if (isa<wk::ValueType>(type)) {
      if (auto encoding = packedType(type)) {
        auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
        auto sizes = shape(type);
        if (failed(layout) || sizes.empty() || sizes.back() == 0 ||
            (sizes.back() > 0 && sizes.back() % layout->elements()) ||
            llvm::is_contained(sizes, 0)) return {};
        if (llvm::any_of(sizes, [](int64_t size) { return size < 0; }))
          return MemRefType::get({ShapedType::kDynamic}, IntegerType::get(type.getContext(), 8));
        sizes.back() = sizes.back() / layout->elements() * layout->bytes();
        if (volume(sizes) > 4096)
          return MemRefType::get({ShapedType::kDynamic}, IntegerType::get(type.getContext(), 8));
        return VectorType::get({volume(sizes)}, IntegerType::get(type.getContext(), 8));
      }
      if (!scalar(type).isIntOrIndexOrFloat()) return {};
      auto dimensions = shape(type);
      if (dimensions.empty()) return scalar(type);
      if (llvm::any_of(dimensions, [](int64_t extent) { return extent < 0; }))
        return MemRefType::get({ShapedType::kDynamic}, scalar(type));
      int64_t count = 1;
      for (int64_t dimension : dimensions) {
        if (dimension <= 0) return {};
        if (count > 4096 / dimension)
          return MemRefType::get({ShapedType::kDynamic}, scalar(type));
        count *= dimension;
      }
      return VectorType::get({count}, scalar(type));
    }
    if (type.isIntOrIndexOrFloat()) return scalar(type);
    return {};
  }
  struct Memory {
    Value base, offset, guard;
    SmallVector<Value> extents, strides, origins, limits;
    Value storageMapped, storageMap, fieldElements, recordBytes, recordColumns, interleaveRows;
  };
  DenseMap<Value, Memory> memories;
  DenseMap<Value, Memory> indexedSnapshots;
  llvm::StringMap<Value> dimensions;
  DenseMap<Value, Value> masks;
  DenseMap<Value, SmallVector<Value>> valueOrigins;
  struct Point {
    wk::DomainType domain;
    Value base, active, partition;
  };
  SmallVector<Point> activePoints;
  DenseMap<Value, Point> points;

  LogicalResult artifactBuilder(wk::KernelOp kernel, unsigned argument, wk::EncodingType encoding) {
    auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
    if (failed(layout)) return unsupported(kernel, "x86 derived artifact has no closed interleave layout");
    Location loc = kernel.getLoc();
    std::string prefix = kernel.getSymName().str() + "_" + cast<StringAttr>(kernel.getArgNames()[argument]).getValue().str();
    Type index = b.getIndexType(), pointer = LLVM::LLVMPointerType::get(b.getContext());
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    auto size = func::FuncOp::create(b, loc, prefix + "_packed_size", b.getFunctionType({index, index}, {index}));
    Block *body = size.addEntryBlock();
    b.setInsertionPointToStart(body);
    Value m = body->getArgument(0), k = body->getArgument(1), zero = indexConstant(loc, 0);
    Value rows = indexConstant(loc, layout->rows), elements = indexConstant(loc, layout->elements());
    Value invalid = arith::OrIOp::create(b, loc,
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, k, zero),
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, arith::RemUIOp::create(b, loc, k, elements), zero));
    Value groups = arith::DivUIOp::create(b, loc,
        arith::AddIOp::create(b, loc, m, indexConstant(loc, layout->rows - 1)), rows);
    Value bytes = arith::MulIOp::create(b, loc,
        arith::MulIOp::create(b, loc, groups, arith::DivUIOp::create(b, loc, k, elements)),
        indexConstant(loc, layout->rows * layout->bytes()));
    func::ReturnOp::create(b, loc, arith::SelectOp::create(b, loc, invalid, zero, bytes).getResult());

    b.setInsertionPointToEnd(module.getBody());
    auto pack = func::FuncOp::create(b, loc, prefix + "_pack", b.getFunctionType({pointer, pointer, index, index}, {}));
    body = pack.addEntryBlock(); b.setInsertionPointToStart(body);
    Value source = body->getArgument(0), destination = body->getArgument(1);
    m = body->getArgument(2); k = body->getArgument(3);
    zero = indexConstant(loc, 0); Value one = indexConstant(loc, 1);
    rows = indexConstant(loc, layout->rows); elements = indexConstant(loc, layout->elements());
    Value blocks = arith::DivUIOp::create(b, loc, k, elements);
    groups = arith::DivUIOp::create(b, loc, arith::AddIOp::create(b, loc, m, indexConstant(loc, layout->rows - 1)), rows);
    Value totalRows = arith::MulIOp::create(b, loc, groups, rows);
    auto rowLoop = scf::ForOp::create(b, loc, zero, totalRows, one);
    b.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar(), group = arith::DivUIOp::create(b, loc, row, rows);
    Value within = arith::RemUIOp::create(b, loc, row, rows);
    Value live = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, row, m);
    auto blockLoop = scf::ForOp::create(b, loc, zero, blocks, one);
    b.setInsertionPointToStart(blockLoop.getBody());
    Value block = blockLoop.getInductionVar();
    Value sourceBase = arith::MulIOp::create(b, loc,
        arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, row, blocks), block), indexConstant(loc, layout->bytes()));
    Value targetBase = arith::MulIOp::create(b, loc,
        arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, group, blocks), block), indexConstant(loc, layout->bytes() * layout->rows));
    for (const auto &span : layout->spans) {
      auto byteLoop = scf::ForOp::create(b, loc, zero, indexConstant(loc, span.byteCount), one);
      b.setInsertionPointToStart(byteLoop.getBody());
      Value byte = byteLoop.getInductionVar(), chunk = indexConstant(loc, span.chunkBytes);
      Value sourceOffset = arith::AddIOp::create(b, loc, sourceBase,
          arith::AddIOp::create(b, loc, indexConstant(loc, span.byteOffset), byte));
      Value targetOffset = arith::AddIOp::create(b, loc, targetBase,
          arith::AddIOp::create(b, loc, indexConstant(loc, span.byteOffset * layout->rows),
              arith::AddIOp::create(b, loc,
                  arith::MulIOp::create(b, loc, arith::DivUIOp::create(b, loc, byte, chunk), indexConstant(loc, span.chunkBytes * layout->rows)),
                  arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, within, chunk), arith::RemUIOp::create(b, loc, byte, chunk)))));
      sourceOffset = arith::IndexCastOp::create(b, loc, b.getI64Type(), sourceOffset);
      targetOffset = arith::IndexCastOp::create(b, loc, b.getI64Type(), targetOffset);
      Value value = access(loc, source, sourceOffset, live, constant(loc, b.getI8Type(), b.getI64IntegerAttr(0)), b.getI8Type(), true, false);
      access(loc, destination, targetOffset, constant(loc, b.getI1Type(), b.getBoolAttr(true)), value, b.getI8Type(), false, false);
      b.setInsertionPointAfter(byteLoop);
    }
    b.setInsertionPointAfter(rowLoop); func::ReturnOp::create(b, loc);
    return success();
  }

  Value operation(Location loc, StringRef name, Type type, ValueRange operands,
                  ArrayRef<NamedAttribute> attributes = {}) {
    OperationState state(loc, name);
    state.addTypes(type);
    state.addOperands(operands);
    state.addAttributes(attributes);
    return b.create(state)->getResult(0);
  }
  Value constant(Location loc, Type type, Attribute value) {
    Type elem = scalar(type);
    if (auto integer = dyn_cast<IntegerAttr>(value)) {
      if (isa<FloatType>(elem)) value = b.getFloatAttr(elem, integer.getValue().getSExtValue());
      else {
        unsigned width = elem.isIndex() ? 64 : cast<IntegerType>(elem).getWidth();
        auto source = dyn_cast<IntegerType>(integer.getType());
        APInt bits = source && (source.isUnsigned() || source.getWidth() == 1) ?
            integer.getValue().zextOrTrunc(width) : integer.getValue().sextOrTrunc(width);
        value = b.getIntegerAttr(elem, bits);
      }
    } else if (auto number = dyn_cast<FloatAttr>(value)) {
      value = b.getFloatAttr(elem, number.getValueAsDouble());
    }
    if (auto vector = dyn_cast<VectorType>(type)) value = DenseElementsAttr::get(vector, value);
    return arith::ConstantOp::create(b, loc, type, cast<TypedAttr>(value));
  }
  Value zero(Location loc, Type type) { return constant(loc, type, b.getI64IntegerAttr(0)); }
  Value permute(Location loc, Value value, ArrayRef<int64_t> indices, Type result) {
    if (scalar(value.getType()).isBF16()) {
      Type bitsType = b.getI16Type(), resultBits = b.getI16Type();
      if (auto vector = dyn_cast<VectorType>(value.getType())) bitsType = vector.clone(bitsType);
      if (auto vector = dyn_cast<VectorType>(result)) resultBits = vector.clone(resultBits);
      Value bits = operation(loc, "arith.bitcast", bitsType, value);
      bits = permute(loc, bits, indices, resultBits);
      return operation(loc, "arith.bitcast", result, bits);
    }
    if (!isa<VectorType>(value.getType()))
      return vector::BroadcastOp::create(b, loc, result, value);
    if (indices.size() == 1 && !isa<VectorType>(result))
      return vector::ExtractOp::create(b, loc, value, ArrayRef<int64_t>{indices.front()});
    return vector::ShuffleOp::create(b, loc, value, value, indices);
  }
  Value broadcast(Location loc, Value value, Type from, Type to) {
    Type result = shape(to).empty() ? scalar(value.getType()) :
        Type(VectorType::get({volume(shape(to))}, scalar(value.getType())));
    if (from == to || (!isa<VectorType>(result) && !isa<VectorType>(value.getType()))) return value;
    auto inputShape = shape(from), outputShape = shape(to), inputAxes = axes(from), outputAxes = axes(to);
    if (inputShape.empty()) return vector::BroadcastOp::create(b, loc, result, value);
    SmallVector<int64_t> indices;
    for (int64_t linear = 0, count = volume(outputShape); linear < count; ++linear) {
      int64_t selected = 0, stride = 1;
      for (int64_t d = inputShape.size() - 1; d >= 0; --d) {
        auto found = llvm::find(outputAxes, inputAxes[d]);
        assert(found != outputAxes.end() && "canonical broadcast must preserve non-singleton axes");
        int64_t position = found - outputAxes.begin();
        int64_t coordinate = (linear / volume(ArrayRef(outputShape).drop_front(position + 1))) % outputShape[position];
        selected += (inputShape[d] == 1 ? 0 : coordinate) * stride;
        stride *= inputShape[d];
      }
      indices.push_back(selected);
    }
    return permute(loc, value, indices, result);
  }
  Value mapped(Value value) { return values.lookup(value); }
  Value aligned(Value value, Type type, Location loc) { return broadcast(loc, mapped(value), value.getType(), type); }
  Value valid(Value value, Location loc) {
    if (auto found = masks.find(value); found != masks.end()) return found->second;
    return constant(loc, physicalMask(value.getType()), b.getBoolAttr(true));
  }
  Type physicalMask(Type type) const {
    if (isa<wk::ValueType>(type) && isa_and_nonnull<MemRefType>(physicalType(type)))
      return MemRefType::get({ShapedType::kDynamic}, IntegerType::get(type.getContext(), 8));
    if (packedType(type) && isa<wk::ValueType>(type))
      return cast<VectorType>(physicalType(type)).clone(IntegerType::get(type.getContext(), 1));
    if (auto shaped = dyn_cast<wk::ValueType>(type))
      return physicalType(wk::ValueType::get(type.getContext(), IntegerType::get(type.getContext(), 1), shaped.getShape(), shaped.getAxisIds()));
    return IntegerType::get(type.getContext(), 1);
  }
  Value validAs(Value value, Type domain, Location loc) {
    return broadcast(loc, valid(value, loc), value.getType(), maskType(domain));
  }
  SmallVector<Value> origins(Value value, Location loc) {
    if (auto found = valueOrigins.find(value); found != valueOrigins.end()) return found->second;
    return SmallVector<Value>(shape(value.getType()).size(), indexConstant(loc, 0));
  }
  bool numeric(Type type) const {
    return isa<wk::ValueType>(type) || type.isIntOrIndexOrFloat();
  }
  SmallVector<Type> carryTypes(Type type) {
    if (isa<wk::PointType>(type)) return SmallVector<Type>(3, b.getIndexType());
    if (memoryType(type)) {
      SmallVector<Type> result{LLVM::LLVMPointerType::get(b.getContext()), b.getIndexType(), b.getI1Type()};
      result.append(shape(type).size() * 4, b.getIndexType());
      llvm::append_range(result, ArrayRef<Type>{b.getI1Type(), LLVM::LLVMPointerType::get(b.getContext()),
          b.getIndexType(), b.getIndexType(), b.getIndexType(), b.getIndexType()});
      return result;
    }
    Type physical = physicalType(type);
    if (!physical) return {};
    SmallVector<Type> result{physical};
    if (numeric(type)) result.push_back(physicalMask(type));
    if (isa<wk::ValueType>(type)) result.append(shape(type).size(), b.getIndexType());
    if (isa<MemRefType>(physical)) result.append(shape(type).size(), b.getIndexType());
    return result;
  }
  SmallVector<Value> carryValues(Value value, Location loc) {
    if (isa<wk::PointType>(value.getType())) {
      auto &point = points.at(value);
      return {point.base, point.active, point.partition};
    }
    if (memoryType(value.getType())) {
      auto &view = memories.at(value);
      SmallVector<Value> result{view.base, view.offset, view.guard};
      llvm::append_range(result, view.extents);
      llvm::append_range(result, view.strides);
      llvm::append_range(result, view.origins);
      llvm::append_range(result, view.limits);
      llvm::append_range(result, ValueRange{view.storageMapped, view.storageMap, view.fieldElements,
          view.recordBytes, view.recordColumns, view.interleaveRows});
      return result;
    }
    SmallVector<Value> result{mapped(value)};
    if (numeric(value.getType())) result.push_back(valid(value, loc));
    if (isa<wk::ValueType>(value.getType())) llvm::append_range(result, origins(value, loc));
    if (isa<MemRefType>(mapped(value).getType())) llvm::append_range(result, arrayExtents(value, loc));
    return result;
  }
  void bindCarry(Value source, ValueRange flat, unsigned &cursor) {
    if (auto type = dyn_cast<wk::PointType>(source.getType())) {
      points[source] = {type.getDomain(), flat[cursor], flat[cursor + 1], flat[cursor + 2]};
      cursor += 3;
    } else if (memoryType(source.getType())) {
      Memory view{flat[cursor++], flat[cursor++], flat[cursor++], {}, {}, {}, {}};
      for (auto *part : {&view.extents, &view.strides, &view.origins, &view.limits})
        for (size_t i = 0; i < shape(source.getType()).size(); ++i) part->push_back(flat[cursor++]);
      for (Value *part : {&view.storageMapped, &view.storageMap, &view.fieldElements,
                         &view.recordBytes, &view.recordColumns, &view.interleaveRows}) *part = flat[cursor++];
      memories[source] = std::move(view);
    } else {
      values.map(source, flat[cursor++]);
      if (numeric(source.getType())) masks[source] = flat[cursor++];
      if (isa<wk::ValueType>(source.getType())) {
        SmallVector<Value> origin;
        for (size_t axis = 0; axis < shape(source.getType()).size(); ++axis) origin.push_back(flat[cursor++]);
        valueOrigins[source] = std::move(origin);
      }
      if (isa<MemRefType>(mapped(source).getType())) {
        SmallVector<Value> extents;
        for (size_t axis = 0; axis < shape(source.getType()).size(); ++axis) extents.push_back(flat[cursor++]);
        arrayShapes[source] = std::move(extents);
      }
    }
  }
  Value castValue(Location loc, Value input, Type from, Type to) {
    Type result = physicalType(to), src = scalar(from), dst = scalar(to);
    if (input.getType() == result) return input;
    if ((src.isIndex() && isa<FloatType>(dst)) || (dst.isIndex() && isa<FloatType>(src))) {
      Type integer = b.getI64Type();
      if (auto vector = dyn_cast<VectorType>(result)) integer = VectorType::get(vector.getShape(), integer);
      if (src.isIndex()) {
        Value bits = operation(loc, "arith.index_castui", integer, input);
        return operation(loc, "arith.uitofp", result, bits);
      }
      Value bits = operation(loc, "arith.fptoui", integer, input);
      return operation(loc, "arith.index_castui", result, bits);
    }
    if (src.isIndex() || dst.isIndex())
      return operation(loc, unsignedType(from) ? "arith.index_castui" : "arith.index_cast", result, input);
    if (isa<IntegerType>(src) && isa<IntegerType>(dst)) {
      auto name = src.getIntOrFloatBitWidth() > dst.getIntOrFloatBitWidth() ? "arith.trunci" :
                  unsignedType(from) || src.isInteger(1) ? "arith.extui" : "arith.extsi";
      return operation(loc, name, result, input);
    }
    if (src.isBF16() && (dst.isF32() || dst.isF64())) {
      // BF16 is exactly the high half of an IEEE f32. Keep widening in
      // integer SIMD operations: legalizing large BF16 shuffles through
      // x86's floating registers can otherwise introduce truncation libcalls.
      Type narrow = b.getI16Type(), wide = b.getI32Type(), fp32 = b.getF32Type();
      if (auto vector = dyn_cast<VectorType>(result)) {
        narrow = vector.clone(narrow);
        wide = vector.clone(wide);
        fp32 = vector.clone(fp32);
      }
      Value bits = operation(loc, "arith.bitcast", narrow, input);
      bits = operation(loc, "arith.extui", wide, bits);
      bits = operation(loc, "arith.shli", wide,
                       {bits, constant(loc, wide, b.getI32IntegerAttr(16))});
      Value value = operation(loc, "arith.bitcast", fp32, bits);
      return dst.isF32() ? value : operation(loc, "arith.extf", result, value);
    }
    if (src.isF32() && dst.isBF16()) return bfloatNarrow(loc, input, result, false);
    if (src.isF16() && (dst.isF32() || dst.isF64()) && !isa<VectorType>(input.getType()) &&
        target.vectorBits == 512 && !target.hasFeature("avx512fp16")) {
      // A scalar half inserted into an undefined XMM carrier can acquire a
      // false dependency on a live accumulator after register allocation.
      // Define the whole conversion carrier with a word broadcast instead.
      // AVX-512 FP16 has a scalar conversion and needs no vector carrier.
      Value value = widenX86ScalarHalf(module, b, loc, input, target);
      return dst.isF32() ? value : operation(loc, "arith.extf", result, value);
    }
    if (isa<FloatType>(src) && isa<FloatType>(dst))
      return operation(loc, src.getIntOrFloatBitWidth() > dst.getIntOrFloatBitWidth() ? "arith.truncf" : "arith.extf", result, input);
    if (isa<FloatType>(dst))
      return operation(loc, unsignedType(from) || src.isInteger(1) ? "arith.uitofp" : "arith.sitofp", result, input);
    return operation(loc, unsignedType(to) ? "arith.fptoui" : "arith.fptosi", result, input);
  }
  Value halfNarrow(Location loc, Value input, Type result, bool towardZero) {
    int64_t count = isa<VectorType>(result) ? cast<VectorType>(result).getNumElements() : 1;
    int64_t padded = (count + 7) / 8 * 8;
    Type packet = VectorType::get({8}, b.getI16Type());
    Type storage = VectorType::get({padded}, b.getI16Type());
    Value output = zero(loc, storage);
    for (int64_t begin = 0; begin < count; begin += 8) {
      SmallVector<int64_t> indices;
      for (int64_t i = 0; i < 8; ++i) indices.push_back(std::min(begin + i, count - 1));
      Value source = permute(loc, input, indices, VectorType::get({8}, b.getF32Type()));
      Value mode = arith::ConstantIntOp::create(b, loc, towardZero ? 3 : 0, 32);
      Value converted = intrinsicCall(loc, "llvm.x86.vcvtps2ph.256", packet, ValueRange{source, mode});
      output = vector::InsertStridedSliceOp::create(b, loc, converted, output, ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
    }
    SmallVector<int64_t> indices;
    for (int64_t i = 0; i < count; ++i) indices.push_back(i);
    Type bits = isa<VectorType>(result) ? Type(VectorType::get({count}, b.getI16Type())) : b.getI16Type();
    output = permute(loc, output, indices, bits);
    return operation(loc, "arith.bitcast", result, output);
  }
  Value bfloatNarrow(Location loc, Value input, Type result, bool towardZero) {
    Type wide = b.getI32Type(), narrow = b.getI16Type();
    if (auto vector = dyn_cast<VectorType>(result)) {
      wide = vector.clone(wide);
      narrow = vector.clone(narrow);
    }
    auto integer = [&](uint32_t value) { return constant(loc, wide, b.getI32IntegerAttr(value)); };
    Value bits = operation(loc, "arith.bitcast", wide, input);
    Value high = operation(loc, "arith.shrui", wide, {bits, integer(16)});
    Value absolute = operation(loc, "arith.andi", wide, {bits, integer(0x7fffffff)});
    Value nan = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, absolute, integer(0x7f800000));
    Value quiet = operation(loc, "arith.ori", wide, {high, integer(0x40)});
    Value rounded = high;
    if (!towardZero) {
      Value odd = operation(loc, "arith.andi", wide, {high, integer(1)});
      Value increment = operation(loc, "arith.addi", wide, {integer(0x7fff), odd});
      rounded = operation(loc, "arith.shrui", wide,
          {operation(loc, "arith.addi", wide, {bits, increment}), integer(16)});
    }
    rounded = arith::SelectOp::create(b, loc, nan, quiet, rounded);
    Value output = operation(loc, "arith.trunci", narrow, rounded);
    return operation(loc, "arith.bitcast", result, output);
  }
  std::optional<int64_t> stride(Value value, unsigned depth = 0) {
    if (depth > 32) return {};
    if (shape(value.getType()).empty()) return 0;
    if (auto iota = value.getDefiningOp<wk::IotaOp>()) return 1;
    if (auto c = value.getDefiningOp<wk::ConstantOp>()) {
      auto dense = dyn_cast<DenseElementsAttr>(c.getValue());
      return isa<IntegerAttr>(c.getValue()) || (dense && dense.isSplat()) ? std::optional<int64_t>(0) : std::nullopt;
    }
    if (auto fresh = value.getDefiningOp<wk::NewOp>())
      return fresh.getInitial() ? stride(fresh.getInitial(), depth + 1) : std::nullopt;
    if (auto binary = value.getDefiningOp<wk::BinaryOp>()) {
      // A narrow add can wrap before extension to pointer width. It must keep
      // its individual addresses unless the complete index range is proven.
      if (!element(value.getType()).isInteger(64)) return {};
      auto a = stride(binary.getLhs(), depth + 1), c = stride(binary.getRhs(), depth + 1);
      if (a && c && binary.getKind() == "add") return *a + *c;
      if (a && c && binary.getKind() == "sub") return *a - *c;
    }
    // A bounded iota can cross a signedness boundary without wrapping. Do not
    // generalize this to an arbitrary dynamic narrow arithmetic expression.
    Operation *cast = value.getDefiningOp();
    if (cast && isa<wk::CastOp, wk::WidenOp>(cast)) {
      auto src = dyn_cast<IntegerType>(element(cast->getOperand(0).getType()));
      auto dst = dyn_cast<IntegerType>(element(value.getType()));
      Value original = cast->getOperand(0);
      for (unsigned i = 0; i < 8; ++i) {
        if (auto range = original.getDefiningOp<wk::IotaOp>()) {
          if (src && dst && range.getStart() >= 0 &&
              APInt(64, range.getEnd() - 1).getActiveBits() < std::min(src.getWidth(), dst.getWidth()))
            return 1;
          break;
        }
        auto *conversion = original.getDefiningOp();
        if (!conversion || !isa<wk::CastOp, wk::WidenOp>(conversion)) break;
        auto inner = dyn_cast<IntegerType>(element(conversion->getOperand(0).getType()));
        if (!inner || !src || inner.getWidth() > src.getWidth()) break;
        original = conversion->getOperand(0);
      }
    }
    return {};
  }
  Value addressLane(Location loc, Value value, int64_t lane, unsigned depth = 0) {
    auto type = dyn_cast<VectorType>(value.getType());
    if (!type) return value;
    if (depth < 16 && (type.getElementType().isIndex() || isa<IntegerType>(type.getElementType()))) {
      if (auto c = value.getDefiningOp<arith::ConstantOp>())
        if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue()))
          return arith::ConstantOp::create(b, loc, b.getIntegerAttr(type.getElementType(), dense.getValues<APInt>()[lane]));
      if (auto splat = value.getDefiningOp<vector::BroadcastOp>())
        if (!isa<VectorType>(splat.getSource().getType())) return splat.getSource();
      if (auto shuffle = value.getDefiningOp<vector::ShuffleOp>()) {
        int64_t index = shuffle.getMask()[lane], split = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
        if (index >= 0) return index < split ? addressLane(loc, shuffle.getV1(), index, depth + 1) :
                                             addressLane(loc, shuffle.getV2(), index - split, depth + 1);
      }
      Operation *producer = value.getDefiningOp();
      if (producer && isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                          arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                          arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp,
                          arith::DivUIOp, arith::RemUIOp, arith::SelectOp, arith::CmpIOp,
                          arith::IndexCastOp, arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(producer)) {
        SmallVector<Value> inputs;
        for (Value input : producer->getOperands()) inputs.push_back(addressLane(loc, input, lane, depth + 1));
        OperationState state(loc, producer->getName());
        state.addOperands(inputs); state.addTypes(type.getElementType()); state.addAttributes(producer->getAttrs());
        return b.create(state)->getResult(0);
      }
    }
    return vector::ExtractOp::create(b, loc, value, ArrayRef<int64_t>{lane});
  }
  Value access(Location loc, Value base, Value offsets, Value mask, Value data,
               Type domain, bool load, bool contiguous) {
    Type elementType = scalar(domain), pointer = LLVM::LLVMPointerType::get(b.getContext());
    Type vectorType = physicalType(domain);
    bool booleanStorage = elementType.isInteger(1);
    bool bfloatStorage = elementType.isBF16();
    Type storageType = vectorType;
    if (booleanStorage) {
      // CPU bool storage is byte-addressed. LLVM <N x i1> memory would pack
      // eight logical elements into one byte and violate the pointer ABI.
      elementType = b.getI8Type();
      storageType = isa<VectorType>(vectorType) ? Type(VectorType::get(cast<VectorType>(vectorType).getShape(), elementType)) : elementType;
      data = operation(loc, "arith.extui", storageType, data);
    }
    if (bfloatStorage) {
      // Storage is an exact 16-bit payload. LLVM's BF16 gather legalization
      // otherwise goes through scalar float values and may introduce a
      // __truncsfbf2 libcall even when the source only loads and widens.
      elementType = b.getI16Type();
      storageType = isa<VectorType>(vectorType) ? Type(cast<VectorType>(vectorType).clone(elementType)) : elementType;
      data = operation(loc, "arith.bitcast", storageType, data);
    }
    Value result;
    if (!isa<VectorType>(vectorType)) {
      auto branch = scf::IfOp::create(b, loc, load ? TypeRange{storageType} : TypeRange{}, mask, true);
      { OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
        Value address = LLVM::GEPOp::create(b, loc, pointer, elementType, base, ValueRange{offsets});
        if (load) {
          Value value = LLVM::LoadOp::create(b, loc, elementType, address, 1);
          scf::YieldOp::create(b, loc, value);
        } else LLVM::StoreOp::create(b, loc, data, address, 1);
        b.setInsertionPointToStart(&branch.getElseRegion().front());
        if (load) scf::YieldOp::create(b, loc, data);
      }
      if (load) result = branch.getResult(0);
    } else {
      if (contiguous) {
        // A native row transfer needs one scalar address. Keep scalar address
        // arithmetic scalar instead of making vector addresses and extracting
        // them back into GPRs for each row. Integer widths/wrap stay unchanged.
        Value first = addressLane(loc, offsets, 0);
        Value address = LLVM::GEPOp::create(b, loc, pointer, elementType, base, ValueRange{first});
        auto storageVector = cast<VectorType>(storageType);
        int64_t packetBits = storageVector.getNumElements() * storageVector.getElementTypeBitWidth();
        if (load && packetBits <= 64 && llvm::isPowerOf2_64(packetBits) &&
            storageVector.getElementTypeBitWidth() < 32) {
          // Byte/word codebook rows often occupy one native scalar word.
          // Narrow masked vector loads otherwise expand to a branch per byte.
          // A complete active row has exactly this byte span; partial rows keep
          // the original masked operation, with no wider speculative access.
          Value full = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, mask);
          auto choice = scf::IfOp::create(b, loc, TypeRange{storageType}, full, true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&choice.getThenRegion().front());
            Type wordType = b.getIntegerType(packetBits);
            Value word = LLVM::LoadOp::create(b, loc, wordType, address, 1);
            scf::YieldOp::create(b, loc, operation(loc, "llvm.bitcast", storageType, word));
            b.setInsertionPointToStart(&choice.getElseRegion().front());
            Value masked = LLVM::MaskedLoadOp::create(b, loc, storageType,
                address, mask, data, b.getI32IntegerAttr(1), UnitAttr{});
            scf::YieldOp::create(b, loc, masked);
          }
          result = choice.getResult(0);
        } else if (load) result = LLVM::MaskedLoadOp::create(b, loc, storageType, address, mask, data, b.getI32IntegerAttr(1), UnitAttr{});
        else LLVM::MaskedStoreOp::create(b, loc, data, address, mask, b.getI32IntegerAttr(1));
      } else if (load && !elementType.isIndex() && elementType.getIntOrFloatBitWidth() < 32) {
        // AVX2/AVX-512 gather has no byte/halfword form. Make those indexed
        // loads explicit at fixed result positions, instead of asking LLVM to
        // legalize a very wide masked gather with a dynamic vector accumulator.
        result = data;
        for (int64_t lane = 0; lane < cast<VectorType>(storageType).getNumElements(); ++lane) {
          auto extract = [&](Value value) -> Value {
            return vector::ExtractOp::create(b, loc, value, ArrayRef<int64_t>{lane});
          };
          Value selected = access(loc, base, addressLane(loc, offsets, lane), extract(mask), extract(data), elementType, true, false);
          result = vector::InsertOp::create(b, loc, selected, result, ArrayRef<int64_t>{lane});
        }
      } else {
        Type pointers = VectorType::get({cast<VectorType>(vectorType).getNumElements()}, pointer);
        Value addresses = LLVM::GEPOp::create(b, loc, pointers, elementType, base, ValueRange{offsets});
        if (load) result = LLVM::masked_gather::create(b, loc, storageType, addresses, mask, ValueRange{data}, b.getI32IntegerAttr(1));
        else LLVM::masked_scatter::create(b, loc, data, addresses, mask, b.getI32IntegerAttr(1));
      }
    }
    if (load) {
      if (booleanStorage)
        result = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, result, zero(loc, storageType));
      if (bfloatStorage) result = operation(loc, "arith.bitcast", vectorType, result);
    }
    return result;
  }
  Value accessRuns(Location loc, Value base, Value offsets, Value mask,
                   Value data, Type domain, bool load, int64_t run) {
    auto vector = dyn_cast<VectorType>(data.getType());
    if (!vector || run == 1 || run == vector.getNumElements())
      return access(loc, base, offsets, mask, data, domain, load,
                    vector && run == vector.getNumElements());
    // A tile can consist of unit-stride rows separated by a dynamic leading
    // stride. Select each proven contiguous run without assuming that the
    // complete flattened tile is contiguous or that its tail is in bounds.
    auto packet = wk::ValueType::get(b.getContext(), element(domain),
        b.getDenseI64ArrayAttr({run}), b.getDenseI64ArrayAttr({axes(domain).back()}));
    Value result = data;
    for (int64_t begin = 0; begin < vector.getNumElements(); begin += run) {
      SmallVector<int64_t> lanes(run);
      std::iota(lanes.begin(), lanes.end(), begin);
      auto part = [&](Value value) {
        return permute(loc, value, lanes,
                       cast<VectorType>(value.getType()).cloneWith({run}, scalar(value.getType())));
      };
      Value value = access(loc, base, part(offsets), part(mask), part(data),
                           packet, load, true);
      if (load) {
        if (vector.getElementType().isBF16()) {
          Value bits = operation(loc, "arith.bitcast", vector.clone(b.getI16Type()), result);
          Value chunk = operation(loc, "arith.bitcast", VectorType::get({run}, b.getI16Type()), value);
          bits = vector::InsertStridedSliceOp::create(b, loc, chunk, bits,
              ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
          result = operation(loc, "arith.bitcast", vector, bits);
        } else result = vector::InsertStridedSliceOp::create(b, loc, value, result,
            ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
      }
    }
    return load ? result : Value();
  }
  Value accessRuntimeRuns(Location loc, Value base, Value offsets, Value mask,
                          Value data, Type domain, bool load, int64_t run, Value stride) {
    APInt constantStride;
    if (matchPattern(stride, m_ConstantInt(&constantStride)))
      return accessRuns(loc, base, offsets, mask, data, domain, load,
                        constantStride.isOne() ? run : 1);
    Value contiguous = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                            stride, indexConstant(loc, 1));
    auto choice = scf::IfOp::create(b, loc, load ? TypeRange{data.getType()} : TypeRange{}, contiguous, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&choice.getThenRegion().front());
      Value result = accessRuns(loc, base, offsets, mask, data, domain, load, run);
      if (load) scf::YieldOp::create(b, loc, result);
      b.setInsertionPointToStart(&choice.getElseRegion().front());
      auto vector = cast<VectorType>(data.getType());
      unsigned payloadBits = vector.getElementType().isIndex() ? 64 : vector.getElementTypeBitWidth();
      // A non-unit stride must not make every row's address vector live
      // across the surrounding computation. Keep the address expansion inside
      // a bounded native row helper when both the full address carrier and
      // payload exceed the target's vector register file. A register-resident
      // payload retains its inline alternate to avoid call-clobber spills.
      // The unit-stride branch stays inline.
      if (vector.getNumElements() > run &&
          llvm::divideCeil(uint64_t(vector.getNumElements()) * payloadBits,
                           uint64_t(target.vectorBits)) > target.vectorRegisters() &&
          llvm::divideCeil(uint64_t(vector.getNumElements()) * 64, uint64_t(target.vectorBits)) > target.vectorRegisters()) {
        auto packet = wk::ValueType::get(b.getContext(), element(domain),
            b.getDenseI64ArrayAttr({run}), b.getDenseI64ArrayAttr({axes(domain).back()}));
        auto payload = VectorType::get({run}, vector.getElementType());
        auto helperPayload = vector.getElementType().isBF16() ? payload.clone(b.getI16Type()) : payload;
        auto predicate = VectorType::get({run}, b.getI1Type());
        std::string elementName;
        llvm::raw_string_ostream stream(elementName);
        vector.getElementType().print(stream);
        std::string name = "__weft_strided_" + std::string(load ? "load_" : "store_") +
                           elementName + "_" + std::to_string(run);
        auto helper = module.lookupSymbol<func::FuncOp>(name);
        if (!helper) {
          OpBuilder::InsertionGuard insertion(b);
          b.setInsertionPointToEnd(module.getBody());
          SmallVector<Type> returns;
          if (load) returns.push_back(helperPayload);
          helper = func::FuncOp::create(b, loc, name, b.getFunctionType(
              {base.getType(), b.getI64Type(), b.getI64Type(), predicate, helperPayload}, returns));
          helper.setPrivate();
          helper->setAttr("weft.x86.cpu", b.getStringAttr(target.cpu));
          helper->setAttr("weft.x86.vector_bits", b.getI32IntegerAttr(target.vectorBits));
          helper->setAttr("weft.x86.vector_registers", b.getI32IntegerAttr(target.vectorRegisters()));
          // This generated body contains only integer address formation and
          // masked SIMD/scalar transfers, so a call cannot alter AMX state.
          helper->setAttr("weft.x86.preserves_tile_state", b.getUnitAttr());
          helper->setAttr("weft.x86.helper", b.getDictionaryAttr({
              b.getNamedAttr("name", b.getStringAttr(name)), b.getNamedAttr("noinline", b.getBoolAttr(true))}));
          auto *body = helper.addEntryBlock();
          b.setInsertionPointToStart(body);
          auto indices = VectorType::get({run}, b.getI64Type());
          SmallVector<APInt> lanes;
          for (int64_t lane = 0; lane < run; ++lane) lanes.emplace_back(64, lane);
          Value coordinates = arith::ConstantOp::create(b, loc, indices, DenseIntElementsAttr::get(indices, lanes));
          Value row = vector::BroadcastOp::create(b, loc, indices, body->getArgument(1));
          Value step = vector::BroadcastOp::create(b, loc, indices, body->getArgument(2));
          Value addresses = arith::AddIOp::create(b, loc, row,
              arith::MulIOp::create(b, loc, coordinates, step));
          Value incoming = body->getArgument(4);
          if (helperPayload != payload) incoming = operation(loc, "arith.bitcast", payload, incoming);
          Value value = access(loc, body->getArgument(0), addresses, body->getArgument(3),
                               incoming, packet, load, false);
          if (load && helperPayload != payload) value = operation(loc, "arith.bitcast", helperPayload, value);
          func::ReturnOp::create(b, loc, load ? ValueRange{value} : ValueRange{});
        }
        Value step = arith::IndexCastOp::create(b, loc, b.getI64Type(), stride);
        result = data;
        for (int64_t begin = 0; begin < vector.getNumElements(); begin += run) {
          SmallVector<int64_t> lanes(run);
          std::iota(lanes.begin(), lanes.end(), begin);
          Value first = addressLane(loc, offsets, begin);
          Value incoming = permute(loc, data, lanes, payload);
          if (helperPayload != payload) incoming = operation(loc, "arith.bitcast", helperPayload, incoming);
          auto call = func::CallOp::create(b, loc, helper, ValueRange{
              base, first, step, permute(loc, mask, lanes, predicate), incoming});
          if (load) {
            if (vector.getElementType().isBF16()) {
              Value bits = operation(loc, "arith.bitcast", vector.clone(b.getI16Type()), result);
              bits = vector::InsertStridedSliceOp::create(b, loc, call.getResult(0), bits,
                  ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
              result = operation(loc, "arith.bitcast", vector, bits);
            } else result = vector::InsertStridedSliceOp::create(b, loc, call.getResult(0), result,
                ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
          }
        }
      } else result = accessRuns(loc, base, offsets, mask, data, domain, load, 1);
      if (load) scf::YieldOp::create(b, loc, result);
    }
    return load ? choice.getResult(0) : Value();
  }
  std::optional<int64_t> modularAxisStep(Value value, int64_t axis, unsigned depth = 0) {
    if (depth > 32) return {};
    auto ids = axes(value.getType()), sizes = shape(value.getType());
    auto found = llvm::find(ids, axis);
    if (found == ids.end() || sizes[found - ids.begin()] == 1) return 0;
    if (value.getDefiningOp<wk::IotaOp>()) return 1;
    if (auto c = value.getDefiningOp<wk::ConstantOp>()) {
      auto dense = dyn_cast<DenseElementsAttr>(c.getValue());
      return isa<IntegerAttr>(c.getValue()) || (dense && dense.isSplat()) ? std::optional<int64_t>(0) : std::nullopt;
    }
    if (auto fresh = value.getDefiningOp<wk::NewOp>())
      return fresh.getInitial() ? modularAxisStep(fresh.getInitial(), axis, depth + 1) : std::nullopt;
    if (auto select = value.getDefiningOp<wk::SelectOp>()) {
      if (select.getTrueValue() == select.getFalseValue()) return modularAxisStep(select.getTrueValue(), axis, depth + 1);
    }
    if (auto reshape = value.getDefiningOp<wk::ReshapeOp>()) {
      auto inputShape = shape(reshape.getInput().getType()), inputAxes = axes(reshape.getInput().getType());
      if (reshape.getOrder() != ArrayRef<int64_t>(inputAxes)) return {};
      SmallVector<int64_t> before, after;
      for (int64_t n : inputShape) if (n != 1) before.push_back(n);
      for (int64_t n : sizes) if (n != 1) after.push_back(n);
      if (before != after) return {};
      unsigned dimension = 0;
      for (auto [id, n] : llvm::zip(ids, sizes)) {
        if (id == axis) break;
        dimension += n != 1;
      }
      for (auto [id, n] : llvm::zip(inputAxes, inputShape))
        if (n != 1) {
          if (!dimension) return modularAxisStep(reshape.getInput(), id, depth + 1);
          --dimension;
        }
      return {};
    }
    auto *op = value.getDefiningOp();
    if (op && isa<wk::CastOp, wk::WidenOp>(op) &&
        isa<IntegerType>(element(value.getType())) && isa<IntegerType>(element(op->getOperand(0).getType())))
      return modularAxisStep(op->getOperand(0), axis, depth + 1);
    if (auto binary = dyn_cast_or_null<wk::BinaryOp>(op)) {
      if (!isa<IntegerType>(element(value.getType()))) return {};
      auto left = modularAxisStep(binary.getLhs(), axis, depth + 1);
      auto right = modularAxisStep(binary.getRhs(), axis, depth + 1);
      if (left && right && binary.getKind() == "add") return *left + *right;
      if (left && right && binary.getKind() == "sub") return *left - *right;
    }
    return {};
  }
  Value accessCheckedRuns(Location loc, Value base, Value offsets, Value mask,
                          Value data, Type domain, bool load, int64_t run) {
    auto vector = cast<VectorType>(data.getType());
    Value result = data;
    for (int64_t begin = 0; begin < vector.getNumElements(); begin += run) {
      SmallVector<int64_t> lanes(run), sequence(run);
      std::iota(lanes.begin(), lanes.end(), begin); std::iota(sequence.begin(), sequence.end(), 0);
      auto part = [&](Value value) {
        return permute(loc, value, lanes, cast<VectorType>(value.getType()).cloneWith({run}, scalar(value.getType())));
      };
      Value addresses = part(offsets), enabled = part(mask), input = part(data);
      Type addressType = addresses.getType();
      Value first = addressLane(loc, addresses, 0);
      Value expected = arith::AddIOp::create(b, loc,
          vector::BroadcastOp::create(b, loc, addressType, first),
          arith::ConstantOp::create(b, loc, addressType, DenseIntElementsAttr::get(cast<ShapedType>(addressType), sequence)));
      Value equal = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, addresses, expected);
      Value contiguous = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, equal);
      auto packet = wk::ValueType::get(b.getContext(), element(domain),
          b.getDenseI64ArrayAttr({run}), b.getDenseI64ArrayAttr({axes(domain).back()}));
      auto branch = scf::IfOp::create(b, loc, load ? TypeRange{input.getType()} : TypeRange{}, contiguous, true);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
        Value selected = access(loc, base, addresses, enabled, input, packet, load, true);
        if (load) scf::YieldOp::create(b, loc, selected);
        b.setInsertionPointToStart(&branch.getElseRegion().front());
        selected = access(loc, base, addresses, enabled, input, packet, load, false);
        if (load) scf::YieldOp::create(b, loc, selected);
      }
      if (load) {
        Value chunk = branch.getResult(0);
        if (vector.getElementType().isBF16()) {
          Value bits = operation(loc, "arith.bitcast", vector.clone(b.getI16Type()), result);
          Value partBits = operation(loc, "arith.bitcast", VectorType::get({run}, b.getI16Type()), chunk);
          bits = vector::InsertStridedSliceOp::create(b, loc, partBits, bits, ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
          result = operation(loc, "arith.bitcast", vector, bits);
        } else result = vector::InsertStridedSliceOp::create(b, loc, chunk, result, ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
      }
    }
    return load ? result : Value();
  }
  bool pointerWidthArithmetic(Value value, unsigned depth = 0) {
    if (depth > 32) return false;
    if (shape(value.getType()).empty()) return true;
    if (auto iota = value.getDefiningOp<wk::IotaOp>()) {
      auto type = dyn_cast<IntegerType>(element(value.getType()));
      return type && iota.getStart() >= 0 && iota.getEnd() > iota.getStart() &&
          APInt(64, iota.getEnd() - 1).getActiveBits() < type.getWidth();
    }
    if (value.getDefiningOp<wk::ConstantOp>()) return stride(value) == std::optional<int64_t>(0);
    if (auto fresh = value.getDefiningOp<wk::NewOp>())
      return fresh.getInitial() && pointerWidthArithmetic(fresh.getInitial(), depth + 1);
    auto type = dyn_cast<IntegerType>(element(value.getType()));
    auto *op = value.getDefiningOp();
    if (op && isa<wk::CastOp, wk::WidenOp>(op) && stride(value) == std::optional<int64_t>(1)) return true;
    if (!op || !type || type.getWidth() != 64) return false;
    if (auto select = dyn_cast<wk::SelectOp>(op))
      return select.getTrueValue() == select.getFalseValue() && pointerWidthArithmetic(select.getTrueValue(), depth + 1);
    return llvm::all_of(op->getOperands(), [&](Value operand) { return pointerWidthArithmetic(operand, depth + 1); });
  }
  LogicalResult memory(Operation &op, bool load) {
    Location loc = op.getLoc();
    Type domain = load ? op.getResult(0).getType() : op.getOperand(3).getType();
    Value offsets = aligned(op.getOperand(1), domain, loc);
    Value mask = aligned(op.getOperand(2), maskType(domain), loc);
    Value data = aligned(op.getOperand(3), domain, loc);
    Value logical = arith::AndIOp::create(b, loc, validAs(op.getOperand(1), domain, loc), validAs(op.getOperand(2), domain, loc));
    mask = arith::AndIOp::create(b, loc, mask, logical);
    if (!load) mask = arith::AndIOp::create(b, loc, mask, validAs(op.getOperand(3), domain, loc));
    auto linear = stride(op.getOperand(1));
    auto sizes = shape(domain);
    bool contiguous = linear && *linear == 1 && sizes.size() == 1;
    auto modular = sizes.empty() ? std::nullopt : modularAxisStep(op.getOperand(1), axes(domain).back());
    int64_t run = sizes.empty() ? 1 : std::min<int64_t>(sizes.back(), target.vectorBits / scalar(domain).getIntOrFloatBitWidth());
    // Narrow arithmetic may wrap before pointer-width extension. Prove actual
    // addresses at the access, including every lane; otherwise preserve the
    // original gather/scatter and mask. No alignment or no-wrap promise is added.
    Value result;
    if (!contiguous && scalar(offsets.getType()).isInteger(64) &&
        modular == std::optional<int64_t>(1) && run > 1 && sizes.back() % run == 0) {
      result = pointerWidthArithmetic(op.getOperand(1)) ?
          accessRuns(loc, mapped(op.getOperand(0)), offsets, mask, data, domain, load, run) :
          accessCheckedRuns(loc, mapped(op.getOperand(0)), offsets, mask, data, domain, load, run);
    } else result = access(loc, mapped(op.getOperand(0)), offsets, mask, data, domain, load, contiguous);
    // Ordinary Triton masked loads return a defined 'other' value. Their mask
    // is not a logical tail domain, unlike a View admit.
    if (load) {
      values.map(op.getResult(0), result);
      Value supplied = arith::SelectOp::create(b, loc, mask,
          constant(loc, physicalType(maskType(domain)), b.getBoolAttr(true)), validAs(op.getOperand(3), domain, loc));
      masks[op.getResult(0)] = arith::AndIOp::create(b, loc, logical, supplied);
    }
    return success();
  }
  Type maskType(Type domain) {
    if (auto shaped = dyn_cast<wk::ValueType>(domain))
      return wk::ValueType::get(b.getContext(), b.getI1Type(), shaped.getShape(), shaped.getAxisIds());
    return b.getI1Type();
  }
  Value indexConstant(Location loc, int64_t value) { return arith::ConstantIndexOp::create(b, loc, value); }
  Value remaining(Location loc, Value extent, Value offset) {
    Value within = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, offset, extent);
    return arith::SelectOp::create(b, loc, within, arith::SubIOp::create(b, loc, extent, offset), indexConstant(loc, 0));
  }
  LogicalResult slice(wk::SliceOp op) {
    Memory result = memories.at(op.getBase());
    SmallVector<Value> sizes, strides, origins, limits;
    unsigned cursor = 0;
    for (size_t axis = 0; axis < result.extents.size(); ++axis) {
      StringRef selector = axis < op.getSelectors().size() ? cast<StringAttr>(op.getSelectors()[axis]).getValue() : "all";
      Value extent = result.extents[axis], origin = result.origins[axis], limit = result.limits[axis];
      bool keep = selector == "all" || selector == "domain";
      if (selector != "all") {
        Value source = op.getIndices()[cursor++], offset;
        if (selector == "index") {
          offset = mapped(source);
          Value within = arith::CmpIOp::create(b, op.getLoc(), arith::CmpIPredicate::ult, offset, limit);
          result.guard = arith::AndIOp::create(b, op.getLoc(), result.guard, within);
        } else if (selector == "domain" || selector == "group_index") {
          auto found = points.find(source);
          if (found == points.end()) return op.emitError("x86 slice lost its Level point");
          Point point = found->second;
          Value coordinate = selector == "domain" ? point.base : Value(arith::DivUIOp::create(b, op.getLoc(), point.base, point.partition));
          offset = arith::SubIOp::create(b, op.getLoc(), coordinate, origin);
          if (keep) {
            extent = arith::MinUIOp::create(b, op.getLoc(), point.active, remaining(op.getLoc(), extent, offset));
            limit = arith::MinUIOp::create(b, op.getLoc(), extent, remaining(op.getLoc(), limit, offset));
            origin = coordinate;
          } else {
            Value within = arith::CmpIOp::create(b, op.getLoc(), arith::CmpIPredicate::ult, offset, limit);
            result.guard = arith::AndIOp::create(b, op.getLoc(), result.guard, within);
          }
        } else return unsupported(op, "unsupported x86 View selector: ") << selector;
        result.offset = arith::AddIOp::create(b, op.getLoc(), result.offset,
            arith::MulIOp::create(b, op.getLoc(), offset, result.strides[axis]));
      }
      if (keep) { sizes.push_back(extent); strides.push_back(result.strides[axis]); origins.push_back(origin); limits.push_back(limit); }
    }
    result.extents = std::move(sizes); result.strides = std::move(strides); result.origins = std::move(origins);
    result.limits = std::move(limits);
    memories[op.getResult()] = std::move(result);
    return success();
  }
  LogicalResult subview(wk::SubviewOp op) {
    Memory result = memories.at(op.getBase());
    unsigned cursor = 0;
    SmallVector<Value> offsets;
    for (int64_t offset : op.getOffsets())
      offsets.push_back(offset == -1 ? mapped(op.getDynamicBounds()[cursor++]) : indexConstant(op.getLoc(), offset));
    for (auto [axis, bound] : llvm::enumerate(op.getExtents())) {
      Value extent = bound == -1 ? mapped(op.getDynamicBounds()[cursor++]) : indexConstant(op.getLoc(), bound);
      result.limits[axis] = arith::MinUIOp::create(b, op.getLoc(), extent, remaining(op.getLoc(), result.limits[axis], offsets[axis]));
      // The source extent is an explicit operand, not a replacement ABI size.
      // Keep it separate from the remaining allocation bound used by loads.
      result.extents[axis] = extent;
      result.offset = arith::AddIOp::create(b, op.getLoc(), result.offset,
          arith::MulIOp::create(b, op.getLoc(), offsets[axis], result.strides[axis]));
    }
    memories[op.getResult()] = std::move(result);
    return success();
  }
  void initializeStorage(Memory &view, Location loc) {
    view.storageMapped = constant(loc, b.getI1Type(), b.getBoolAttr(false));
    view.storageMap = LLVM::ZeroOp::create(b, loc, LLVM::LLVMPointerType::get(b.getContext()));
    view.fieldElements = view.recordBytes = view.recordColumns = view.interleaveRows = indexConstant(loc, 1);
  }
  Value storageTable(Location loc, weft::analysis::EncodingBytes &layout, int field = -1, bool persistent = true) {
    unsigned width = field < 0 ? 8 : cast<TypeAttr>(layout.base.getFieldTypes()[field]).getValue().getIntOrFloatBitWidth();
    int64_t count = field < 0 ? layout.bytes() : volume(cast<DenseI64ArrayAttr>(layout.base.getFieldShapes()[field]).asArrayRef());
    int64_t slots = (width + 7) / 8 + 1;
    SmallVector<int64_t> data;
    for (int64_t index = 0; index < count; ++index) {
      SmallVector<weft::analysis::StorageByteFragment> fragments;
      if (field < 0) {
        auto [byte, chunk] = layout.address(index);
        fragments.push_back({byte, 0, 255, 0, chunk});
      } else fragments = layout.fragments(field, index, persistent);
      assert(fragments.size() <= static_cast<size_t>(slots));
      fragments.resize(slots);
      for (auto fragment : fragments)
        llvm::append_range(data, ArrayRef<int64_t>{fragment.byte, fragment.shift, fragment.mask, fragment.resultShift, fragment.chunk});
    }
    LLVM::GlobalOp table;
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToEnd(module.getBody());
      unsigned sequence = 0;
      std::string name;
      do { name = "__weft_storage_" + std::to_string(sequence++); }
      while (SymbolTable::lookupSymbolIn(module, name));
      auto type = VectorType::get({static_cast<int64_t>(data.size())}, b.getI64Type());
      table = LLVM::GlobalOp::create(b, loc, type, true, LLVM::Linkage::Private, name,
          DenseIntElementsAttr::get(type, data), 8);
    }
    return LLVM::AddressOfOp::create(b, loc, table);
  }
  Value storageAccess(Location loc, const Memory &view, Value offsets, Value active,
                      Value data, Type domain, bool load) {
    Type resultType = data.getType();
    unsigned width = scalar(domain).getIntOrFloatBitWidth();
    int64_t slots = (width + 7) / 8 + 1;
    auto scalarAccess = [&](Value offset, Value mask, Value payload) -> Value {
      auto as64 = [&](Value value) { return castValue(loc, value, value.getType(), b.getI64Type()); };
      Value linear = as64(offset), fieldSize = as64(view.fieldElements), columns = as64(view.recordColumns);
      Value bytes = as64(view.recordBytes), rows = as64(view.interleaveRows);
      Value record = arith::DivUIOp::create(b, loc, linear, fieldSize);
      Value entry = arith::RemUIOp::create(b, loc, linear, fieldSize);
      Value row = arith::DivUIOp::create(b, loc, record, columns);
      Value column = arith::RemUIOp::create(b, loc, record, columns);
      Value cohort = arith::DivUIOp::create(b, loc, row, rows);
      Value rowWithin = arith::RemUIOp::create(b, loc, row, rows);
      Value base = arith::MulIOp::create(b, loc,
          arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, cohort, columns), column),
          arith::MulIOp::create(b, loc, bytes, rows));
      Type bitsType = b.getIntegerType(width);
      Value bits = load ? zero(loc, bitsType) : payload;
      if (!load && isa<FloatType>(scalar(domain))) bits = operation(loc, "arith.bitcast", bitsType, bits);
      if (width != 64) bits = operation(loc, "arith.extui", b.getI64Type(), bits);
      for (int64_t part = 0; part < slots; ++part) {
        SmallVector<Value> mapping;
        for (int64_t component = 0; component < 5; ++component) {
          Value address = arith::AddIOp::create(b, loc,
              arith::MulIOp::create(b, loc, entry, constant(loc, b.getI64Type(), b.getI64IntegerAttr(slots * 5))),
              constant(loc, b.getI64Type(), b.getI64IntegerAttr(part * 5 + component)));
          mapping.push_back(access(loc, view.storageMap, address, mask, zero(loc, b.getI64Type()), b.getI64Type(), true, false));
        }
        Value address = arith::AddIOp::create(b, loc, base,
            arith::AddIOp::create(b, loc, mapping[0], arith::MulIOp::create(b, loc, rowWithin, mapping[4])));
        Value enabled = arith::AndIOp::create(b, loc, mask,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, mapping[2], zero(loc, b.getI64Type())));
        Value byte = access(loc, view.base, address, enabled, zero(loc, b.getI8Type()), b.getI8Type(), true, false);
        Value wide = operation(loc, "arith.extui", b.getI64Type(), byte);
        if (load) {
          Value fragment = arith::AndIOp::create(b, loc,
              arith::ShRUIOp::create(b, loc, wide, mapping[1]), mapping[2]);
          bits = arith::OrIOp::create(b, loc, bits, arith::ShLIOp::create(b, loc, fragment, mapping[3]));
        } else {
          Value clear = arith::XOrIOp::create(b, loc,
              arith::ShLIOp::create(b, loc, mapping[2], mapping[1]), constant(loc, b.getI64Type(), b.getI64IntegerAttr(-1)));
          Value fragment = arith::AndIOp::create(b, loc,
              arith::ShRUIOp::create(b, loc, bits, mapping[3]), mapping[2]);
          Value update = arith::OrIOp::create(b, loc, arith::AndIOp::create(b, loc, wide, clear),
              arith::ShLIOp::create(b, loc, fragment, mapping[1]));
          access(loc, view.base, address, enabled, operation(loc, "arith.trunci", b.getI8Type(), update), b.getI8Type(), false, false);
        }
      }
      if (width != 64) bits = operation(loc, "arith.trunci", bitsType, bits);
      return isa<FloatType>(scalar(domain)) ? operation(loc, "arith.bitcast", scalar(domain), bits) : bits;
    };
    if (!isa<VectorType>(resultType)) return scalarAccess(offsets, active, data);
    // Sequential byte updates preserve other fields and siblings sharing a
    // storage byte. A scatter of independent read-modify-writes would lose bits.
    Value begin = indexConstant(loc, 0), end = indexConstant(loc, cast<VectorType>(resultType).getNumElements());
    auto loop = scf::ForOp::create(b, loc, begin, end, indexConstant(loc, 1),
        load ? ValueRange{zero(loc, resultType)} : ValueRange{});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value lane = arith::IndexCastOp::create(b, loc, b.getI64Type(), loop.getInductionVar());
      auto extract = [&](Value value) { return operation(loc, "llvm.extractelement", scalar(value.getType()), {value, lane}); };
      Value selected = scalarAccess(extract(offsets), extract(active), extract(data));
      if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back())) loop.getBody()->back().erase();
      b.setInsertionPointToEnd(loop.getBody());
      if (load) {
        Value result = operation(loc, "llvm.insertelement", resultType, {loop.getRegionIterArgs()[0], selected, lane});
        scf::YieldOp::create(b, loc, result);
      } else scf::YieldOp::create(b, loc);
    }
    return load ? loop.getResult(0) : Value();
  }
  LogicalResult field(wk::FieldOp op) {
    auto layout = weft::analysis::encodingBytes(sourceModule, packedType(op.getOwner().getType()));
    if (failed(layout)) return unsupported(op, "x86 field has no complete declared storage mapping");
    auto found = llvm::find(layout->base.getFieldNames(), b.getStringAttr(op.getName()));
    if (found == layout->base.getFieldNames().end()) return op.emitError("field is absent from its Encoding");
    unsigned fieldIndex = found - layout->base.getFieldNames().begin();
    auto fieldShape = cast<DenseI64ArrayAttr>(layout->base.getFieldShapes()[fieldIndex]).asArrayRef();
    int64_t fieldElements = volume(fieldShape);
    if (memoryType(op.getOwner().getType())) {
      Memory view = memories.at(op.getOwner());
      Value recordOrigin = view.origins.back();
      auto fieldOffset = layout->affineFieldOffset(fieldIndex);
      auto fieldType = cast<TypeAttr>(layout->base.getFieldTypes()[fieldIndex]).getValue();
      int64_t recordStride = fieldOffset ? layout->bytes() / (fieldType.getIntOrFloatBitWidth() / 8) : fieldElements;
      view.guard = b.createOrFold<arith::AndIOp>(op.getLoc(), view.guard,
          b.createOrFold<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::uge,
              view.limits.back(), indexConstant(op.getLoc(), layout->elements())));
      view.offset = b.createOrFold<arith::MulIOp>(op.getLoc(),
          b.createOrFold<arith::DivUIOp>(op.getLoc(), view.offset, indexConstant(op.getLoc(), layout->elements())),
          indexConstant(op.getLoc(), recordStride));
      if (fieldOffset) view.offset = b.createOrFold<arith::AddIOp>(op.getLoc(), view.offset,
          indexConstant(op.getLoc(), *fieldOffset));
      for (auto *part : {&view.extents, &view.strides, &view.origins, &view.limits}) part->pop_back();
      for (Value &stride : view.strides)
        stride = b.createOrFold<arith::MulIOp>(op.getLoc(),
            b.createOrFold<arith::DivUIOp>(op.getLoc(), stride, indexConstant(op.getLoc(), layout->elements())),
            indexConstant(op.getLoc(), recordStride));
      for (auto [i, extent] : llvm::enumerate(fieldShape)) {
        view.extents.push_back(indexConstant(op.getLoc(), extent));
        view.limits.push_back(view.extents.back());
        view.strides.push_back(indexConstant(op.getLoc(), volume(fieldShape.drop_front(i + 1))));
        view.origins.push_back(i + 1 == fieldShape.size() && layout->elements() % extent == 0 ?
            Value(arith::DivUIOp::create(b, op.getLoc(), recordOrigin, indexConstant(op.getLoc(), layout->elements() / extent))) :
            indexConstant(op.getLoc(), 0));
      }
      view.storageMapped = constant(op.getLoc(), b.getI1Type(), b.getBoolAttr(!fieldOffset));
      view.storageMap = storageTable(op.getLoc(), *layout, fieldIndex);
      view.fieldElements = indexConstant(op.getLoc(), fieldElements);
      memories[op.getResult()] = std::move(view);
      return success();
    }
    Type resultType = physicalType(op.getResult().getType()), mask = physicalMask(op.getResult().getType());
    unsigned width = scalar(op.getResult().getType()).getIntOrFloatBitWidth();
    auto finish = [&](Value data, Value active) {
      values.map(op.getResult(), data); masks[op.getResult()] = active;
      auto origin = origins(op.getOwner(), op.getLoc());
      Value recordOrigin = origin.back(); origin.pop_back();
      for (auto [i, extent] : llvm::enumerate(fieldShape))
        origin.push_back(i + 1 == fieldShape.size() && layout->elements() % extent == 0 ?
            Value(arith::DivUIOp::create(b, op.getLoc(), recordOrigin, indexConstant(op.getLoc(), layout->elements() / extent))) :
            indexConstant(op.getLoc(), 0));
      valueOrigins[op.getResult()] = std::move(origin);
      return success();
    };
    if (target.predicateRegisters() && width == 1 && fieldElements % 8 == 0 && isa<VectorType>(resultType)) {
      auto first = layout->fragments(fieldIndex, 0, false);
      bool packedBits = first.size() == 1;
      int64_t start = packedBits ? first.front().byte : 0;
      for (int64_t bit = 0; packedBits && bit < fieldElements; ++bit) {
        auto pieces = layout->fragments(fieldIndex, bit, false);
        packedBits = pieces.size() == 1 && pieces[0].byte == start + bit / 8 &&
            pieces[0].shift == bit % 8 && pieces[0].mask == 1 && pieces[0].resultShift == 0;
      }
      if (packedBits) {
        int64_t count = cast<VectorType>(resultType).getNumElements();
        SmallVector<int64_t> bytes, predicates;
        for (int64_t bit = 0; bit < count; ++bit) {
          int64_t offset = bit / fieldElements * layout->bytes() + start + bit % fieldElements / 8;
          if (bit % 8 == 0) bytes.push_back(offset);
          predicates.push_back(offset);
        }
        Value source = permute(op.getLoc(), mapped(op.getOwner()), bytes, VectorType::get({count / 8}, b.getI8Type()));
        Value bits = operation(op.getLoc(), "llvm.bitcast", resultType, source);
        Value active = permute(op.getLoc(), valid(op.getOwner(), op.getLoc()), predicates, mask);
        return finish(bits, active);
      }
    }
    // Byte fragments need only the destination field width. AVX-512BW has
    // per-word variable shifts; AVX2 uses dword shifts. Keep these native shift
    // carriers instead of expanding every narrow field to i64 lanes.
    unsigned carrierBits = target.vectorBits == 512 ? 16 : 32;
    while (carrierBits < width) carrierBits *= 2;
    Type carrier = b.getIntegerType(carrierBits);
    Type bitsType = isa<VectorType>(resultType) ? Type(cast<VectorType>(resultType).clone(carrier)) : carrier;
    Value result = zero(op.getLoc(), bitsType), active = constant(op.getLoc(), mask, b.getBoolAttr(true));
    for (unsigned part = 0; part < (width + 7) / 8 + 1; ++part) {
      SmallVector<int64_t> indices;
      SmallVector<APInt> shifts, fields, outputShifts, required;
      for (int64_t lane = 0; lane < volume(shape(op.getResult().getType())); ++lane) {
        auto fragments = layout->fragments(fieldIndex, lane % fieldElements, false);
        weft::analysis::StorageByteFragment fragment = part < fragments.size() ? fragments[part] : weft::analysis::StorageByteFragment{};
        indices.push_back(lane / fieldElements * layout->bytes() + fragment.byte);
        shifts.emplace_back(carrierBits, fragment.shift); fields.emplace_back(carrierBits, fragment.mask);
        outputShifts.emplace_back(carrierBits, fragment.resultShift); required.emplace_back(1, fragment.mask != 0);
      }
      auto literals = [&](Type type, ArrayRef<APInt> elements) -> Value {
        return isa<VectorType>(type) ? Value(arith::ConstantOp::create(b, op.getLoc(), type, DenseIntElementsAttr::get(cast<ShapedType>(type), elements))) :
            constant(op.getLoc(), type, b.getIntegerAttr(type, elements.front()));
      };
      Type bytes = isa<VectorType>(resultType) ? Type(cast<VectorType>(resultType).clone(b.getI8Type())) : b.getI8Type();
      Value data = operation(op.getLoc(), "arith.extui", bitsType, permute(op.getLoc(), mapped(op.getOwner()), indices, bytes));
      Value fragment = arith::AndIOp::create(b, op.getLoc(), arith::ShRUIOp::create(b, op.getLoc(), data, literals(bitsType, shifts)), literals(bitsType, fields));
      result = arith::OrIOp::create(b, op.getLoc(), result, arith::ShLIOp::create(b, op.getLoc(), fragment, literals(bitsType, outputShifts)));
      Value validPart = arith::SelectOp::create(b, op.getLoc(), literals(mask, required),
          permute(op.getLoc(), valid(op.getOwner(), op.getLoc()), indices, mask), constant(op.getLoc(), mask, b.getBoolAttr(true)));
      active = arith::AndIOp::create(b, op.getLoc(), active, validPart);
    }
    Type integer = isa<VectorType>(resultType) ? Type(cast<VectorType>(resultType).clone(b.getIntegerType(width))) : b.getIntegerType(width);
    if (width != carrierBits) result = operation(op.getLoc(), "arith.trunci", integer, result);
    if (isa<FloatType>(scalar(op.getResult().getType()))) result = operation(op.getLoc(), "arith.bitcast", resultType, result);
    return finish(result, active);
  }
  bool knownPowerOfTwoMultiple(Value value, int64_t divisor, unsigned depth = 0) {
    if (depth > 16 || !llvm::isPowerOf2_64(divisor)) return false;
    if (divisor == 1) return true;
    APInt number;
    if (matchPattern(value, m_ConstantInt(&number)))
      return number.getBitWidth() <= 64 && (number.getZExtValue() & (divisor - 1)) == 0;
    if (auto add = value.getDefiningOp<arith::AddIOp>())
      return knownPowerOfTwoMultiple(add.getLhs(), divisor, depth + 1) &&
             knownPowerOfTwoMultiple(add.getRhs(), divisor, depth + 1);
    if (auto mul = value.getDefiningOp<arith::MulIOp>())
      return knownPowerOfTwoMultiple(mul.getLhs(), divisor, depth + 1) ||
             knownPowerOfTwoMultiple(mul.getRhs(), divisor, depth + 1);
    auto *definition = value.getDefiningOp();
    if (definition && isa<arith::IndexCastOp, arith::IndexCastUIOp,
                          arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(definition)) {
      auto bits = [](Type type) { return type.isIndex() ? 64u : type.getIntOrFloatBitWidth(); };
      if (llvm::Log2_64(divisor) <= std::min(bits(value.getType()), bits(definition->getOperand(0).getType())))
        return knownPowerOfTwoMultiple(definition->getOperand(0), divisor, depth + 1);
    }
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
      return loop && value == loop.getInductionVar() &&
             knownPowerOfTwoMultiple(loop.getLowerBound(), divisor, depth + 1) &&
             knownPowerOfTwoMultiple(loop.getStep(), divisor, depth + 1);
    }
    return false;
  }
  Value accessDeclaredField(Location loc, Value region, const Memory &view,
                            Value offsets, Value active, Type domain, Value input = {}) {
    Type resultType = physicalType(domain);
    Value descriptor = region;
    int64_t windowOffset = 0, windowElements = 0;
    Value translatedOffset;
    if (auto window = region.getDefiningOp<wk::SubviewOp>()) {
      auto baseShape = shape(window.getBase().getType());
      if (window.getOffsets().size() != baseShape.size() || baseShape.empty()) return {};
      for (size_t axis = 0; axis + 1 < baseShape.size(); ++axis)
        if (window.getOffsets()[axis] != 0 || window.getExtents()[axis] != baseShape[axis]) return {};
      windowOffset = window.getOffsets().back();
      windowElements = window.getExtents().back();
      if (windowOffset == -1 && window.getDynamicBounds().size() == 1) {
        translatedOffset = mapped(window.getDynamicBounds().front());
        windowOffset = 0;
      }
      if (windowOffset < 0 || windowElements <= 0) return {};
      descriptor = window.getBase();
    } else if (auto slice = region.getDefiningOp<wk::SliceOp>()) {
      auto selectors = slice.getSelectors().getValue();
      if (selectors.empty() || slice.getIndices().size() != 1 ||
          cast<StringAttr>(selectors.back()).getValue() != "index" ||
          !llvm::all_of(selectors.drop_back(), [](Attribute selector) {
            return cast<StringAttr>(selector).getValue() == "all";
          })) return {};
      APInt coordinate;
      Value index = mapped(slice.getIndices().front());
      if (matchPattern(index, m_ConstantInt(&coordinate))) {
        if (coordinate.isNegative() || coordinate.getActiveBits() > 63) return {};
        windowOffset = coordinate.getSExtValue();
      } else translatedOffset = index;
      windowElements = 1;
      descriptor = slice.getBase();
    }
    auto field = descriptor.getDefiningOp<wk::FieldOp>();
    auto vectorType = dyn_cast<VectorType>(resultType);
    if (!field || !vectorType || !memoryType(field.getOwner().getType())) return {};
    auto layout = weft::analysis::encodingBytes(sourceModule, packedType(field.getOwner().getType()));
    if (failed(layout)) return {};
    auto found = llvm::find(layout->base.getFieldNames(), b.getStringAttr(field.getName()));
    if (found == layout->base.getFieldNames().end()) return {};
    unsigned fieldIndex = found - layout->base.getFieldNames().begin();
    if (layout->affineFieldOffset(fieldIndex)) return {};
    auto fieldShape = cast<DenseI64ArrayAttr>(layout->base.getFieldShapes()[fieldIndex]).asArrayRef();
    int64_t elements = volume(fieldShape);
    if (windowElements) {
      if (fieldShape.size() != 1 || windowOffset > elements || windowElements > elements - windowOffset) return {};
    } else windowElements = elements;
    std::optional<std::pair<int64_t, int64_t>> translation;
    if (translatedOffset) {
      if (!knownPowerOfTwoMultiple(translatedOffset, windowElements)) return {};
      translation = layout->translatedWindow(fieldIndex, windowElements);
      if (!translation) return {};
    }
    int64_t count = vectorType.getNumElements();
    if (!windowElements || count % windowElements) return {};
    // A field descriptor covers complete records. A static subview can demand
    // a smaller field window without changing those record identities. Its
    // fragments determine exactly which bytes are read; outer record validity
    // and the original complete-record guard still mask every byte.
    SmallVector<SmallVector<weft::analysis::StorageByteFragment>> fragments;
    SmallVector<int64_t> bytes;
    unsigned parts = 0;
    for (int64_t lane = 0; lane < windowElements; ++lane) {
      fragments.push_back(layout->fragments(fieldIndex, windowOffset + lane, false));
      parts = std::max(parts, static_cast<unsigned>(fragments.back().size()));
      for (auto fragment : fragments.back()) bytes.push_back(fragment.byte);
    }
    llvm::sort(bytes);
    bytes.erase(std::unique(bytes.begin(), bytes.end()), bytes.end());
    int64_t records = count / windowElements, byteCount = bytes.size();
    if (!byteCount || records * byteCount > 4096) return {};
    bool partialCohort = layout->rows != 1 && records < layout->rows;
    if (layout->rows != 1 && !partialCohort && records % layout->rows) return {};
    int64_t spanRows = partialCohort ? records : layout->rows;
    SmallVector<int64_t> elementForByte;
    if (input) {
      // A complete byte field can be permuted directly into physical storage.
      // Subbyte/shared-byte writes retain ordered read-modify-write semantics.
      if (!vectorType.getElementType().isInteger(8) || byteCount != windowElements) return {};
      elementForByte.resize(byteCount, -1);
      for (int64_t lane = 0; lane < windowElements; ++lane) {
        if (fragments[lane].size() != 1) return {};
        auto fragment = fragments[lane].front();
        if (fragment.shift || fragment.resultShift || fragment.mask != 255) return {};
        int64_t position = llvm::lower_bound(bytes, fragment.byte) - bytes.begin();
        if (elementForByte[position] != -1) return {};
        elementForByte[position] = lane;
      }
    }
    OpBuilder::InsertionGuard insertion(b);
    auto translatedBase = [&](Value base, int64_t stride) -> Value {
      if (!translatedOffset || !stride) return base;
      Value offset = translatedOffset;
      if (offset.getType().isIndex())
        offset = arith::IndexCastUIOp::create(b, loc, b.getI64Type(), offset);
      else if (!offset.getType().isInteger(64))
        offset = arith::ExtUIOp::create(b, loc, b.getI64Type(), offset);
      auto number = [&](int64_t n) { return constant(loc, b.getI64Type(), b.getI64IntegerAttr(n)); };
      Value windows = arith::DivUIOp::create(b, loc, offset, number(windowElements));
      return arith::AddIOp::create(b, loc, base, arith::MulIOp::create(b, loc, windows, number(stride)));
    };
    scf::IfOp cohortChoice;
    auto rawType = VectorType::get({records * byteCount}, b.getI8Type());
    Value raw = zero(loc, rawType);
    if (layout->rows != 1) {
      auto number = [&](int64_t n) { return constant(loc, b.getI64Type(), b.getI64IntegerAttr(n)); };
      Value first = vector::ExtractOp::create(b, loc, offsets, ArrayRef<int64_t>{0});
      Value firstRecord = arith::DivUIOp::create(b, loc, first, number(elements));
      Value columns = arith::IndexCastOp::create(b, loc, b.getI64Type(), view.recordColumns);
      Value row = arith::DivUIOp::create(b, loc, firstRecord, columns);
      Value column = arith::RemUIOp::create(b, loc, firstRecord, columns);
      bool directCohort = false;
      Value cohortOwner = field.getOwner();
      int64_t cohortRowOffset = 0;
      int64_t cohortColumnOffset = 0;
      while (auto window = cohortOwner.getDefiningOp<wk::SubviewOp>()) {
        auto parent = shape(window.getBase().getType());
        auto start = window.getOffsets(), extent = window.getExtents();
        if (parent.size() == 2 && start.size() == 2 && extent.size() == 2 &&
            start[0] >= 0 && extent[0] > 0 && extent[0] <= parent[0] &&
            start[0] <= parent[0] - extent[0] && start[0] % spanRows == 0 &&
            extent[0] % spanRows == 0 && start[1] >= 0 && extent[1] > 0 &&
            extent[1] <= parent[1] && start[1] <= parent[1] - extent[1] &&
            start[1] % layout->elements() == 0 && extent[1] % layout->elements() == 0 &&
            cohortRowOffset <= INT64_MAX - start[0] &&
            cohortColumnOffset <= INT64_MAX - start[1] / layout->elements()) {
          cohortOwner = window.getBase();
          cohortRowOffset += start[0];
          cohortColumnOffset += start[1] / layout->elements();
        } else break;
      }
      // A root two-dimensional View selected by two domain points retains
      // independent row/block coordinates. Reuse those facts instead of
      // dividing a flattened runtime-stride address back into coordinates.
      if (auto selection = cohortOwner.getDefiningOp<wk::SliceOp>()) {
        auto root = dyn_cast<BlockArgument>(selection.getBase());
        auto windowShape = shape(field.getOwner().getType());
        if (root && isa<wk::KernelOp>(root.getOwner()->getParentOp()) &&
            shape(root.getType()).size() == 2 && selection.getSelectors().size() == 2 &&
            windowShape.size() == 2 && windowShape.back() == layout->elements() &&
            cast<StringAttr>(selection.getSelectors()[0]).getValue() == "domain" &&
            cast<StringAttr>(selection.getSelectors()[1]).getValue() == "domain") {
          auto rowPoint = points.find(selection.getIndices()[0]);
          auto blockPoint = points.find(selection.getIndices()[1]);
          if (rowPoint != points.end() && blockPoint != points.end() &&
              layout->rows % spanRows == 0 &&
              knownPowerOfTwoMultiple(rowPoint->second.base, spanRows)) {
            row = arith::IndexCastOp::create(b, loc, b.getI64Type(), rowPoint->second.base);
            row = arith::AddIOp::create(b, loc, row, number(cohortRowOffset));
            column = arith::DivUIOp::create(b, loc,
                arith::IndexCastOp::create(b, loc, b.getI64Type(), blockPoint->second.base),
                number(layout->elements()));
            column = arith::AddIOp::create(b, loc, column, number(cohortColumnOffset));
            directCohort = true;
          }
        }
      }
      Value rowInCohort = arith::RemUIOp::create(b, loc, row, number(layout->rows));
      Value aligned = arith::CmpIOp::create(b, loc,
          partialCohort ? arith::CmpIPredicate::ule : arith::CmpIPredicate::eq,
          rowInCohort, number(partialCohort ? layout->rows - records : 0));
      // Recognize an actual complete cohort or a contained row slice. Do
      // not infer alignment or row stride just from the field's vector shape.
      for (int64_t record = 1; !directCohort && record < records; ++record) {
        Value offset = vector::ExtractOp::create(b, loc, offsets, ArrayRef<int64_t>{record * windowElements});
        Value expected = arith::AddIOp::create(b, loc, first,
            arith::MulIOp::create(b, loc, columns, number(record * elements)));
        aligned = arith::AndIOp::create(b, loc, aligned,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, offset, expected));
      }
      Value base = arith::MulIOp::create(b, loc,
          arith::AddIOp::create(b, loc, column, arith::MulIOp::create(b, loc, columns,
              arith::DivUIOp::create(b, loc, row, number(layout->rows)))), number(layout->rows * layout->bytes()));
      if (translation) base = translatedBase(base, translation->second);
      if (!directCohort) {
        cohortChoice = scf::IfOp::create(b, loc, input ? TypeRange{} : TypeRange{resultType}, aligned, true);
        b.setInsertionPointToStart(&cohortChoice.getThenRegion().front());
      }
      SmallVector<std::pair<int64_t, int64_t>> order;
      SmallVector<int64_t> chunks;
      for (int64_t byte : bytes) chunks.push_back(layout->address(byte).second);
      for (int64_t record = 0; record < spanRows; ++record)
        for (int64_t i = 0; i < byteCount; ++i) {
          auto [byte, chunk] = layout->address(bytes[i]);
          order.emplace_back(byte + record * chunk, record * byteCount + i);
        }
      llvm::sort(order);
      auto cohortType = VectorType::get({spanRows * byteCount}, b.getI8Type());
      for (int64_t cohort = 0; cohort < records / spanRows; ++cohort) {
        Value cohortBase = arith::AddIOp::create(b, loc, base,
            arith::MulIOp::create(b, loc, columns, number(cohort * layout->rows * layout->bytes())));
        Value packed = zero(loc, cohortType);
        SmallVector<int64_t> inverse(spanRows * byteCount);
        for (int64_t begin = 0; begin < static_cast<int64_t>(order.size());) {
          int64_t end = begin + 1;
          while (end < static_cast<int64_t>(order.size()) && order[end].first == order[end - 1].first + 1 &&
                 (!partialCohort || chunks[order[end].second % byteCount] == chunks[order[begin].second % byteCount])) ++end;
          auto packetType = VectorType::get({end - begin}, b.getI8Type());
          auto addressType = packetType.clone(b.getI64Type());
          SmallVector<int64_t> positions, predicates;
          for (int64_t i = begin; i < end; ++i) {
            positions.push_back(order[i].first);
            predicates.push_back((cohort * spanRows + order[i].second / byteCount) * windowElements +
                                 (input ? elementForByte[order[i].second % byteCount] : 0));
            inverse[order[i].second] = i;
          }
          Value packetBase = cohortBase;
          if (partialCohort)
            packetBase = arith::AddIOp::create(b, loc, packetBase,
                arith::MulIOp::create(b, loc, rowInCohort, number(chunks[order[begin].second % byteCount])));
          Value address = arith::AddIOp::create(b, loc,
              arith::ConstantOp::create(b, loc, addressType, DenseIntElementsAttr::get(addressType, positions)),
              vector::BroadcastOp::create(b, loc, addressType, packetBase));
          Value mask = permute(loc, active, predicates, packetType.clone(b.getI1Type()));
          auto packetDomain = wk::ValueType::get(b.getContext(), b.getIntegerType(8, IntegerType::Unsigned),
              b.getDenseI64ArrayAttr({end - begin}), b.getDenseI64ArrayAttr({axes(region.getType()).back()}));
          Value payload = input ? permute(loc, input, predicates, packetType) : zero(loc, packetType);
          Value packet = access(loc, view.base, address, mask, payload, packetDomain, !input, true);
          if (!input) packed = vector::InsertStridedSliceOp::create(b, loc, packet, packed,
              ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
          begin = end;
        }
        if (!input) {
          Value cohortRaw = permute(loc, packed, inverse, cohortType);
          raw = vector::InsertStridedSliceOp::create(b, loc, cohortRaw, raw,
              ArrayRef<int64_t>{cohort * spanRows * byteCount}, ArrayRef<int64_t>{1});
        }
      }
    } else for (int64_t record = 0; record < records; ++record) {
      Value linear = vector::ExtractOp::create(b, loc, offsets, ArrayRef<int64_t>{record * windowElements});
      Value live = vector::ExtractOp::create(b, loc, active, ArrayRef<int64_t>{record * windowElements});
      Value base = arith::MulIOp::create(b, loc,
          arith::DivUIOp::create(b, loc, linear, constant(loc, b.getI64Type(), b.getI64IntegerAttr(elements))),
          constant(loc, b.getI64Type(), b.getI64IntegerAttr(layout->bytes())));
      if (translation) base = translatedBase(base, translation->first);
      for (int64_t begin = 0; begin < byteCount;) {
        int64_t end = begin + 1;
        while (end < byteCount && bytes[end] == bytes[end - 1] + 1) ++end;
        auto packetType = VectorType::get({end - begin}, b.getI8Type());
        auto addressType = packetType.clone(b.getI64Type());
        SmallVector<int64_t> coordinates(bytes.begin() + begin, bytes.begin() + end);
        Value positions = arith::ConstantOp::create(b, loc, addressType,
            DenseIntElementsAttr::get(addressType, coordinates));
        Value address = arith::AddIOp::create(b, loc, positions,
            vector::BroadcastOp::create(b, loc, addressType, base));
        SmallVector<int64_t> selected;
        if (input)
          for (int64_t byte = begin; byte < end; ++byte)
            selected.push_back(record * windowElements + elementForByte[byte]);
        Value mask = input ? permute(loc, active, selected, packetType.clone(b.getI1Type())) :
                            Value(vector::BroadcastOp::create(b, loc, packetType.clone(b.getI1Type()), live));
        auto packetDomain = wk::ValueType::get(b.getContext(),
            b.getIntegerType(8, IntegerType::Unsigned), b.getDenseI64ArrayAttr({end - begin}),
            b.getDenseI64ArrayAttr({axes(region.getType()).back()}));
        Value payload = input ? permute(loc, input, selected, packetType) : zero(loc, packetType);
        Value packet = access(loc, view.base, address, mask, payload, packetDomain, !input, true);
        if (!input) raw = vector::InsertStridedSliceOp::create(b, loc, packet, raw,
            ArrayRef<int64_t>{record * byteCount + begin}, ArrayRef<int64_t>{1});
        begin = end;
      }
    }
    if (input) {
      if (cohortChoice) {
        b.setInsertionPointToStart(&cohortChoice.getElseRegion().front());
        storageAccess(loc, view, offsets, active, input, domain, false);
      }
      return input;
    }
    unsigned width = vectorType.getElementType().getIntOrFloatBitWidth();
    unsigned carrierBits = target.vectorBits == 512 ? 16 : 32;
    while (carrierBits < width) carrierBits *= 2;
    auto carrierType = vectorType.clone(b.getIntegerType(carrierBits));
    Value result = zero(loc, carrierType);
    for (unsigned part = 0; part < parts; ++part) {
      SmallVector<int64_t> indices;
      SmallVector<APInt> shifts, fields, outputShifts;
      for (int64_t lane = 0; lane < count; ++lane) {
        auto &pieces = fragments[lane % windowElements];
        auto fragment = part < pieces.size() ? pieces[part] : weft::analysis::StorageByteFragment{};
        int64_t index = fragment.mask ? llvm::lower_bound(bytes, fragment.byte) - bytes.begin() : 0;
        indices.push_back(lane / windowElements * byteCount + index);
        shifts.emplace_back(carrierBits, fragment.shift);
        fields.emplace_back(carrierBits, fragment.mask);
        outputShifts.emplace_back(carrierBits, fragment.resultShift);
      }
      auto literals = [&](ArrayRef<APInt> values) -> Value {
        return arith::ConstantOp::create(b, loc, carrierType, DenseIntElementsAttr::get(carrierType, values));
      };
      Value data = operation(loc, "arith.extui", carrierType,
          permute(loc, raw, indices, vectorType.clone(b.getI8Type())));
      Value fragment = arith::AndIOp::create(b, loc,
          arith::ShRUIOp::create(b, loc, data, literals(shifts)), literals(fields));
      result = arith::OrIOp::create(b, loc, result,
          arith::ShLIOp::create(b, loc, fragment, literals(outputShifts)));
    }
    auto integerType = vectorType.clone(b.getIntegerType(width));
    if (width != carrierBits) result = operation(loc, "arith.trunci", integerType, result);
    if (isa<FloatType>(vectorType.getElementType())) result = operation(loc, "arith.bitcast", resultType, result);
    if (cohortChoice) {
      scf::YieldOp::create(b, loc, result);
      b.setInsertionPointToStart(&cohortChoice.getElseRegion().front());
      Value general = storageAccess(loc, view, offsets, active, zero(loc, resultType), domain, true);
      scf::YieldOp::create(b, loc, general);
      return cohortChoice.getResult(0);
    }
    return result;
  }
  LogicalResult viewAccess(Operation &op, Value region, Value input = {}) {
    Memory view = memories.at(region);
    Location loc = op.getLoc();
    Type domain = input ? input.getType() : op.getResult(0).getType();
    Value recordLimit;
    int64_t recordBytes = 1;
    if (auto encoding = packedType(domain)) {
      auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
      if (failed(layout)) return unsupported(op, "x86 encoded Value has no storage declaration");
      auto sizes = shape(domain);
      recordBytes = layout->bytes();
      recordLimit = b.createOrFold<arith::DivUIOp>(loc, view.limits.back(),
                                                 indexConstant(loc, layout->elements()));
      sizes.back() = sizes.back() / layout->elements() * layout->bytes();
      domain = wk::ValueType::get(b.getContext(), b.getIntegerType(8, IntegerType::Unsigned), b.getDenseI64ArrayAttr(sizes), b.getDenseI64ArrayAttr(axes(domain)));
      auto convert = [&](Value value) { return Value(arith::MulIOp::create(b, loc,
          arith::DivUIOp::create(b, loc, value, indexConstant(loc, layout->elements())), indexConstant(loc, layout->bytes()))); };
      view.offset = convert(view.offset);
      for (size_t axis = 0; axis + 1 < view.strides.size(); ++axis) view.strides[axis] = convert(view.strides[axis]);
      view.extents.back() = convert(view.extents.back()); view.limits.back() = convert(view.limits.back());
      view.origins.back() = convert(view.origins.back());
      // A whole base record is already in byte order. Only a persistent
      // interleave changes those byte addresses; field decoding happens after
      // the snapshot and must not turn a contiguous record load into a scalar
      // storage-table walk.
      view.storageMapped = constant(loc, b.getI1Type(), b.getBoolAttr(layout->rows != 1));
      view.storageMap = storageTable(loc, *layout);
      view.fieldElements = indexConstant(loc, layout->bytes());
    }
    auto sizes = shape(domain), ids = axes(domain), memoryAxes = axes(region.getType());
    Type offsetsType = b.getIndexType(), mask = b.getI1Type();
    if (!sizes.empty()) { offsetsType = VectorType::get({volume(sizes)}, b.getIndexType()); mask = VectorType::get({volume(sizes)}, b.getI1Type()); }
    auto splat = [&](Value value, Type type) -> Value {
      return isa<VectorType>(type) ? b.createOrFold<vector::BroadcastOp>(loc, type, value) : value;
    };
    Value offsets = splat(view.offset, offsetsType), active = splat(view.guard, mask);
    for (size_t axis = 0; axis < memoryAxes.size(); ++axis) {
      auto found = llvm::find(ids, memoryAxes[axis]);
      if (found == ids.end()) return op.emitError("x86 View access lost a logical axis");
      size_t position = found - ids.begin();
      SmallVector<APInt> coordinates;
      for (int64_t lane = 0; lane < volume(sizes); ++lane)
        coordinates.emplace_back(64, lane / volume(ArrayRef(sizes).drop_front(position + 1)) % sizes[position]);
      Value index = arith::ConstantOp::create(b, loc, offsetsType,
          DenseElementsAttr::get(cast<VectorType>(offsetsType), coordinates));
      offsets = arith::AddIOp::create(b, loc, offsets, arith::MulIOp::create(b, loc, index, splat(view.strides[axis], offsetsType)));
      // Validity is separable by logical axis. All bytes of an encoded record
      // share one predicate: byte < completeRecords * recordBytes iff
      // byte / recordBytes < completeRecords. Compare only unique coordinates
      // and then project predicates, rather than repeating pointer-width
      // comparisons for every byte and every other axis of the tile.
      bool records = recordLimit && axis + 1 == memoryAxes.size();
      int64_t group = records ? recordBytes : 1;
      int64_t count = sizes[position] / group;
      Value limit = records ? recordLimit : view.limits[axis];
      Value inside;
      if (count == 1) {
        inside = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             indexConstant(loc, 0), limit);
        inside = splat(inside, mask);
      } else {
        auto axisType = VectorType::get({count}, b.getIndexType());
        SmallVector<APInt> unique;
        for (int64_t coordinate = 0; coordinate < count; ++coordinate)
          unique.emplace_back(64, coordinate);
        Value axisIndex = arith::ConstantOp::create(b, loc, axisType,
                                                   DenseElementsAttr::get(axisType, unique));
        inside = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             axisIndex, splat(limit, axisType));
        SmallVector<int64_t> projection;
        for (int64_t lane = 0; lane < volume(sizes); ++lane)
          projection.push_back(lane / volume(ArrayRef(sizes).drop_front(position + 1)) % sizes[position] / group);
        inside = permute(loc, inside, projection, mask);
      }
      active = b.createOrFold<arith::AndIOp>(loc, active, inside);
    }
    // Prove the maximal unit-stride suffix from the actual storage relation.
    // Outer axes may have gaps without losing the contiguous inner packets.
    int64_t contiguousRun = 1;
    for (int64_t axis = sizes.size() - 1; memoryAxes == ids && axis >= 0; --axis) {
      auto step = view.strides[axis].getDefiningOp<arith::ConstantIndexOp>();
      if (sizes[axis] != 1 && (!step || step.value() != contiguousRun)) break;
      contiguousRun *= sizes[axis];
    }
    if (input) active = arith::AndIOp::create(b, loc, active, valid(input, loc));
    Type addressType = sizes.empty() ? Type(b.getI64Type()) : Type(VectorType::get({volume(sizes)}, b.getI64Type()));
    offsets = arith::IndexCastOp::create(b, loc, addressType, offsets);
    Value data = input ? mapped(input) : zero(loc, physicalType(domain));
    if (Value result = accessDeclaredField(loc, region, view, offsets, active, domain, input ? data : Value())) {
      if (!input) {
        values.map(op.getResult(0), result); masks[op.getResult(0)] = active;
        valueOrigins[op.getResult(0)] = memories.at(region).origins;
      }
      return success();
    }
    auto branch = scf::IfOp::create(b, loc, input ? TypeRange{} : TypeRange{data.getType()}, view.storageMapped, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      Value result = storageAccess(loc, view, offsets, active, data, domain, !input);
      if (!input) scf::YieldOp::create(b, loc, result);
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      if (contiguousRun == 1 && memoryAxes == ids && !sizes.empty() && sizes.back() > 1)
        result = accessRuntimeRuns(loc, view.base, offsets, active, data, domain, !input,
                                   sizes.back(), view.strides.back());
      else result = accessRuns(loc, view.base, offsets, active, data, domain, !input, contiguousRun);
      if (!input) scf::YieldOp::create(b, loc, result);
    }
    Value result = input ? Value() : branch.getResult(0);
    if (input)
      if (auto hint = op.getAttrOfType<StringAttr>("cache_hint"); hint && hint.getValue() == "streaming")
        branch.walk([&](LLVM::MaskedStoreOp store) { store->setAttr("weft.x86.streaming", b.getUnitAttr()); });
    if (!input) {
      values.map(op.getResult(0), result); masks[op.getResult(0)] = active;
      valueOrigins[op.getResult(0)] = memories.at(region).origins;
    }
    return success();
  }
  LogicalResult atomic(wk::PointerAtomicOp op) {
    Location loc = op.getLoc();
    Type domain = op.getResult().getType(), type = physicalType(domain), elem = scalar(domain);
    Value logical = arith::AndIOp::create(b, loc, validAs(op.getOffsets(), domain, loc), validAs(op.getMask(), domain, loc));
    logical = arith::AndIOp::create(b, loc, logical, validAs(op.getValue(), domain, loc));
    if (op.getKind() == "cas") logical = arith::AndIOp::create(b, loc, logical, validAs(op.getCompare(), domain, loc));
    auto order = op.getOrdering() == "relaxed" ? LLVM::AtomicOrdering::monotonic :
        op.getOrdering() == "acquire" ? LLVM::AtomicOrdering::acquire :
        op.getOrdering() == "release" ? LLVM::AtomicOrdering::release : LLVM::AtomicOrdering::acq_rel;
    auto lane = [&](Value offset, Value mask, Value update, Value comparison) -> Value {
      auto branch = scf::IfOp::create(b, loc, TypeRange{elem}, mask, true);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      Value pointer = LLVM::GEPOp::create(b, loc, LLVM::LLVMPointerType::get(b.getContext()),
          elem, mapped(op.getBase()), ValueRange{offset});
      Value old;
      unsigned alignment = elem.getIntOrFloatBitWidth() / 8;
      if (op.getKind() == "cas") {
        Type bits = b.getIntegerType(elem.getIntOrFloatBitWidth());
        if (isa<FloatType>(elem)) {
          update = operation(loc, "arith.bitcast", bits, update);
          comparison = operation(loc, "arith.bitcast", bits, comparison);
        }
        auto failureOrder = order == LLVM::AtomicOrdering::release ? LLVM::AtomicOrdering::monotonic :
            order == LLVM::AtomicOrdering::acq_rel ? LLVM::AtomicOrdering::acquire : order;
        Value exchanged = LLVM::AtomicCmpXchgOp::create(b, loc, pointer, comparison, update,
            order, failureOrder, "", alignment);
        old = LLVM::ExtractValueOp::create(b, loc, exchanged, ArrayRef<int64_t>{0});
        if (isa<FloatType>(elem)) old = operation(loc, "arith.bitcast", elem, old);
      } else {
        auto spelling = (op.getKind() == "and" || op.getKind() == "or" || op.getKind() == "xor") ?
            "_" + op.getKind().str() : op.getKind().str();
        auto kind = LLVM::symbolizeAtomicBinOp(spelling);
        // Canonical verification limits this vocabulary to the exact LLVM
        // integer/fadd/xchg operations; a missing spelling is a compiler error.
        if (!kind) return {};
        old = LLVM::AtomicRMWOp::create(b, loc, *kind, pointer, update, order, "", alignment);
      }
      scf::YieldOp::create(b, loc, old);
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      scf::YieldOp::create(b, loc, zero(loc, elem));
      return branch.getResult(0);
    };
    Value result;
    if (auto vector = dyn_cast<VectorType>(type)) {
      Value offsets = aligned(op.getOffsets(), domain, loc), masks = aligned(op.getMask(), maskType(domain), loc);
      masks = arith::AndIOp::create(b, loc, masks, logical);
      Value updates = aligned(op.getValue(), domain, loc), comparisons = aligned(op.getCompare(), domain, loc);
      Value begin = arith::ConstantIndexOp::create(b, loc, 0);
      Value end = arith::ConstantIndexOp::create(b, loc, vector.getNumElements());
      Value step = arith::ConstantIndexOp::create(b, loc, 1);
      auto loop = scf::ForOp::create(b, loc, begin, end, step, ValueRange{zero(loc, type)});
      { OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        Value index = arith::IndexCastOp::create(b, loc, b.getI64Type(), loop.getInductionVar());
        auto extract = [&](Value vector) {
          return operation(loc, "llvm.extractelement", scalar(vector.getType()), {vector, index});
        };
        Value old = lane(extract(offsets), extract(masks), extract(updates), extract(comparisons));
        if (!old) return unsupported(op, "unsupported x86 atomic operation: ") << op.getKind();
        Value output = operation(loc, "llvm.insertelement", type, {loop.getRegionIterArgs()[0], old, index});
        scf::YieldOp::create(b, loc, output);
      }
      result = loop.getResult(0);
    } else {
      Value mask = arith::AndIOp::create(b, loc, mapped(op.getMask()), logical);
      result = lane(mapped(op.getOffsets()), mask, mapped(op.getValue()), mapped(op.getCompare()));
      if (!result) return unsupported(op, "unsupported x86 atomic operation: ") << op.getKind();
    }
    values.map(op.getResult(), result);
    masks[op.getResult()] = logical;
    return success();
  }
  LogicalResult binary(wk::BinaryOp op) {
    auto loc = op.getLoc();
    Type type = op.getResult().getType(), resultType = physicalType(type);
    Value a = aligned(op.getLhs(), type, loc), c = aligned(op.getRhs(), type, loc);
    bool floating = isa<FloatType>(scalar(type)), uns = unsignedType(type);
    StringRef kind = op.getKind(), name;
    if (!floating && (kind == "div" || kind == "rem" || kind == "mod")) {
      Value active = arith::AndIOp::create(b, loc, validAs(op.getLhs(), type, loc), validAs(op.getRhs(), type, loc));
      a = arith::SelectOp::create(b, loc, active, a, zero(loc, resultType));
      c = arith::SelectOp::create(b, loc, active, c, constant(loc, resultType, b.getI64IntegerAttr(1)));
    }
    if (kind == "add") name = floating ? "arith.addf" : "arith.addi";
    else if (kind == "sub") name = floating ? "arith.subf" : "arith.subi";
    else if (kind == "mul") name = floating ? "arith.mulf" : "arith.muli";
    else if (kind == "div") name = floating ? "arith.divf" : uns ? "arith.divui" : "arith.divsi";
    else if (kind == "rem" || kind == "mod") name = floating ? "arith.remf" : uns ? "arith.remui" : "arith.remsi";
    else if (kind == "min") name = floating ? "arith.minnumf" : uns ? "arith.minui" : "arith.minsi";
    else if (kind == "max") name = floating ? "arith.maxnumf" : uns ? "arith.maxui" : "arith.maxsi";
    else if (kind == "minimum") name = "arith.minimumf";
    else if (kind == "maximum") name = "arith.maximumf";
    else if (kind == "and") name = "arith.andi";
    else if (kind == "or") name = "arith.ori";
    else if (kind == "xor") name = "arith.xori";
    else if (kind == "shl") name = "arith.shli";
    else if (kind == "shr") name = uns ? "arith.shrui" : "arith.shrsi";
    else return unsupported(op, "unsupported x86 binary operation: ") << kind;
    if (floating && kind == "add") {
      for (auto pair : {std::make_pair(op.getLhs(), c), std::make_pair(op.getRhs(), a)}) {
        Value source = pair.first;
        // Axis renaming and contiguous reshaping do not round a value. Follow
        // only single-consumer identities within this same source block;
        // casts, permutations and lifecycle boundaries remain barriers.
        while (auto reshape = source.getDefiningOp<wk::ReshapeOp>()) {
          if (!source.hasOneUse() || reshape->getBlock() != op->getBlock() ||
              reshape.getOrder() != ArrayRef<int64_t>(axes(reshape.getInput().getType())) ||
              physicalType(reshape.getInput().getType()) != resultType) break;
          source = reshape.getInput();
        }
        auto mul = source.getDefiningOp<wk::BinaryOp>();
        if (mul && mul.getKind() == "mul" && mul->getBlock() == op->getBlock() &&
            mul.getResult().hasOneUse() && physicalType(mul.getResult().getType()) == resultType &&
            pair.first.getType() == type) {
          Type productType = mul.getResult().getType();
          values.map(op.getResult(), operation(loc, "math.fma", resultType,
              {aligned(mul.getLhs(), productType, loc), aligned(mul.getRhs(), productType, loc), pair.second}));
          return success();
        }
      }
    }
    values.map(op.getResult(), operation(loc, name, resultType, {a, c}));
    return success();
  }
  LogicalResult reduce(wk::ReduceOp op) {
    auto dimensions = shape(op.getInput().getType());
    int64_t axis = op.getAxis(), extent = dimensions[axis];
    Type result = physicalType(op.getResult().getType());
    Type inputType = physicalType(op.getInput().getType()), elem = scalar(op.getInput().getType());
    auto neutral = weft::analysis::reductionIdentity(elem, op.getKind(), unsignedType(op.getInput().getType()));
    if (!neutral) return unsupported(op, "unsupported x86 reduction identity: ") << op.getKind();
    Value identity = constant(op.getLoc(), inputType, neutral);
    Value validity = valid(op.getInput(), op.getLoc());
    Value input = arith::SelectOp::create(b, op.getLoc(), validity, mapped(op.getInput()), identity);
    bool floating = isa<FloatType>(elem), unsignedInput = unsignedType(op.getInput().getType());
    vector::CombiningKind kind;
    if (op.getKind() == "sum" || op.getKind() == "add") kind = vector::CombiningKind::ADD;
    else if (op.getKind() == "max") kind = floating ? vector::CombiningKind::MAXNUMF :
        unsignedInput ? vector::CombiningKind::MAXUI : vector::CombiningKind::MAXSI;
    else if (op.getKind() == "min") kind = floating ? vector::CombiningKind::MINNUMF :
        unsignedInput ? vector::CombiningKind::MINUI : vector::CombiningKind::MINSI;
    else return unsupported(op, "unsupported x86 reduction: ") << op.getKind();
    if (dimensions.size() == 1) {
      values.map(op.getResult(), vector::ReductionOp::create(b, op.getLoc(), kind, input));
      masks[op.getResult()] = vector::ReductionOp::create(b, op.getLoc(), vector::CombiningKind::OR, validity);
      return success();
    }
    if (axis + 1 == int64_t(dimensions.size())) {
      SmallVector<Value> sums, live;
      auto partType = VectorType::get({extent}, elem);
      auto maskPart = VectorType::get({extent}, b.getI1Type());
      for (int64_t row = 0; row < volume(dimensions) / extent; ++row) {
        SmallVector<int64_t> indices(extent);
        std::iota(indices.begin(), indices.end(), row * extent);
        Value part = permute(op.getLoc(), input, indices, partType);
        Value active = permute(op.getLoc(), validity, indices, maskPart);
        sums.push_back(vector::ReductionOp::create(b, op.getLoc(), kind, part));
        live.push_back(vector::ReductionOp::create(b, op.getLoc(), vector::CombiningKind::OR, active));
      }
      values.map(op.getResult(), vector::FromElementsOp::create(b, op.getLoc(), result, sums));
      masks[op.getResult()] = vector::FromElementsOp::create(b, op.getLoc(), physicalType(maskType(op.getResult().getType())), live);
      return success();
    }
    SmallVector<Value> partials;
    SmallVector<Value> validParts;
    int64_t inner = volume(ArrayRef(dimensions).drop_front(axis + 1));
    for (int64_t k = 0; k < extent; ++k) {
      SmallVector<int64_t> indices;
      for (int64_t lane = 0; lane < volume(dimensions) / extent; ++lane)
        indices.push_back((lane / inner * extent + k) * inner + lane % inner);
      partials.push_back(permute(op.getLoc(), input, indices, result));
      validParts.push_back(permute(op.getLoc(), validity, indices, physicalType(maskType(op.getResult().getType()))));
    }
    bool fp = isa<FloatType>(scalar(op.getResult().getType())), uns = unsignedType(op.getResult().getType());
    StringRef name;
    if (op.getKind() == "sum" || op.getKind() == "add") name = fp ? "arith.addf" : "arith.addi";
    else if (op.getKind() == "max") name = fp ? "arith.maxnumf" : uns ? "arith.maxui" : "arith.maxsi";
    else if (op.getKind() == "min") name = fp ? "arith.minnumf" : uns ? "arith.minui" : "arith.minsi";
    else return unsupported(op, "unsupported x86 reduction: ") << op.getKind();
    // Adjacent pairs retain element order for the closed reduction primitive.
    while (partials.size() > 1) {
      SmallVector<Value> next;
      for (size_t i = 0; i < partials.size(); i += 2)
        next.push_back(i + 1 == partials.size() ? partials[i] : operation(op.getLoc(), name, result, {partials[i], partials[i + 1]}));
      partials = std::move(next);
    }
    values.map(op.getResult(), partials.front());
    Value active = validParts.front();
    for (Value part : ArrayRef(validParts).drop_front()) active = arith::OrIOp::create(b, op.getLoc(), active, part);
    masks[op.getResult()] = active;
    return success();
  }
  FailureOr<SmallVector<Value>> combine(Region &region, ValueRange left, ValueRange right,
                                        ArrayRef<int64_t> dimensions, ArrayRef<int64_t> ids) {
    Region scratch;
    IRMapping cloned;
    region.cloneInto(&scratch, cloned);
    auto lift = [&](Type type) -> Type {
      if (dimensions.empty()) return type;
      return wk::ValueType::get(b.getContext(), type,
          b.getDenseI64ArrayAttr(dimensions), b.getDenseI64ArrayAttr(ids));
    };
    IRMapping outerValues = values;
    auto outerMasks = masks;
    auto outerOrigins = valueOrigins;
    auto restore = llvm::scope_exit([&] {
      values = std::move(outerValues); masks = std::move(outerMasks); valueOrigins = std::move(outerOrigins);
    });
    Block &body = scratch.front();
    for (auto [i, arg] : llvm::enumerate(body.getArguments())) {
      arg.setType(lift(arg.getType()));
      values.map(arg, i < left.size() ? left[i] : right[i - left.size()]);
    }
    for (Operation &operation : body.without_terminator()) {
      for (Value result : operation.getResults()) result.setType(lift(result.getType()));
      if (failed(lower(operation))) return failure();
    }
    SmallVector<Value> result;
    for (Value value : body.getTerminator()->getOperands()) result.push_back(mapped(value));
    return result;
  }
  FailureOr<SmallVector<Value>> maskedCombine(Region &region, ValueRange left, ValueRange right,
                                             Value leftValid, Value rightValid, Location loc) {
    SmallVector<Value> output;
    SmallVector<Type> scalarTypes;
    for (Value value : left) { output.push_back(zero(loc, value.getType())); scalarTypes.push_back(scalar(value.getType())); }
    int64_t count = isa<VectorType>(leftValid.getType()) ? cast<VectorType>(leftValid.getType()).getNumElements() : 1;
    for (int64_t lane = 0; lane < count; ++lane) {
      auto extract = [&](Value value) -> Value {
        return isa<VectorType>(value.getType()) ? Value(vector::ExtractOp::create(b, loc, value, ArrayRef<int64_t>{lane})) : value;
      };
      Value aValid = extract(leftValid), bValid = extract(rightValid);
      SmallVector<Value> a, c;
      for (Value value : left) a.push_back(extract(value));
      for (Value value : right) c.push_back(extract(value));
      Value both = arith::AndIOp::create(b, loc, aValid, bValid);
      auto branch = scf::IfOp::create(b, loc, scalarTypes, both, true);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
        auto result = combine(region, a, c, {}, {});
        if (failed(result)) return failure();
        scf::YieldOp::create(b, loc, *result);
        b.setInsertionPointToStart(&branch.getElseRegion().front());
        SmallVector<Value> selected;
        for (auto [lhs, rhs] : llvm::zip(a, c)) selected.push_back(arith::SelectOp::create(b, loc, aValid, lhs, rhs));
        scf::YieldOp::create(b, loc, selected);
      }
      for (auto [i, value] : llvm::enumerate(branch.getResults())) {
        if (isa<VectorType>(output[i].getType())) output[i] = vector::InsertOp::create(b, loc, value, output[i], ArrayRef<int64_t>{lane});
        else output[i] = value;
      }
    }
    return output;
  }
  Value intrinsicCall(Location loc, StringRef name, Type result, ValueRange operands) {
    auto function = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(module, name));
    if (!function) {
      OpBuilder::InsertionGuard guard(b); b.setInsertionPointToEnd(module.getBody());
      function = func::FuncOp::create(b, loc, name, b.getFunctionType(operands.getTypes(), TypeRange{result}));
      function.setPrivate();
      function->setAttr("weft.x86.preserves_tile_state", b.getUnitAttr());
    }
    return func::CallOp::create(b, loc, function, operands).getResult(0);
  }
  Value dynamicRound(Location loc, Value input, bool integerResult = false) {
    Type elementType = scalar(input.getType());
    assert((!integerResult || elementType.isF32()) && "integer dynamic conversion requires f32");
    if (elementType.isF16() || elementType.isBF16()) {
      Type wider = isa<VectorType>(input.getType()) ? Type(cast<VectorType>(input.getType()).clone(b.getF32Type())) : b.getF32Type();
      Value result = dynamicRound(loc, operation(loc, "arith.extf", wider, input));
      return operation(loc, "arith.truncf", input.getType(), result);
    }
    int64_t count = isa<VectorType>(input.getType()) ? cast<VectorType>(input.getType()).getNumElements() : 1;
    unsigned bits = count * elementType.getIntOrFloatBitWidth() <= 128 ? 128 :
                    count * elementType.getIntOrFloatBitWidth() <= 256 ? 256 : target.vectorBits;
    int64_t lanes = bits / elementType.getIntOrFloatBitWidth();
    auto packet = VectorType::get({lanes}, elementType);
    Type outputElement = integerResult ? Type(b.getI32Type()) : elementType;
    auto outputPacket = VectorType::get({lanes}, outputElement);
    auto resultType = VectorType::get({(count + lanes - 1) / lanes * lanes}, outputElement);
    Value output = zero(loc, resultType);
    for (int64_t begin = 0; begin < count; begin += lanes) {
      SmallVector<int64_t> indices;
      for (int64_t lane = 0; lane < lanes; ++lane) indices.push_back(std::min(begin + lane, count - 1));
      Value source = permute(loc, input, indices, packet);
      Value mode = arith::ConstantIntOp::create(b, loc, 4, 32); // MXCSR rounding mode.
      Value rounded;
      if (integerResult) {
        if (bits == 512)
          rounded = intrinsicCall(loc, "llvm.x86.avx512.mask.cvtps2dq.512", outputPacket,
              ValueRange{source, zero(loc, outputPacket), arith::ConstantIntOp::create(b, loc, -1, 16), mode});
        else rounded = intrinsicCall(loc, bits == 256 ? "llvm.x86.avx.cvt.ps2dq.256" : "llvm.x86.sse2.cvtps2dq",
                                     outputPacket, ValueRange{source});
      } else if (bits == 512) {
        Value enabled = arith::ConstantIntOp::create(b, loc, -1, elementType.isF64() ? 8 : 16);
        rounded = intrinsicCall(loc, elementType.isF64() ? "llvm.x86.avx512.mask.rndscale.pd.512" :
            "llvm.x86.avx512.mask.rndscale.ps.512", packet,
            ValueRange{source, mode, zero(loc, packet), enabled, mode});
      } else {
        StringRef name = bits == 256 ?
            (elementType.isF64() ? "llvm.x86.avx.round.pd.256" : "llvm.x86.avx.round.ps.256") :
            (elementType.isF64() ? "llvm.x86.sse41.round.pd" : "llvm.x86.sse41.round.ps");
        rounded = intrinsicCall(loc, name, packet, ValueRange{source, mode});
      }
      output = vector::InsertStridedSliceOp::create(b, loc, rounded, output, ArrayRef<int64_t>{begin}, ArrayRef<int64_t>{1});
    }
    SmallVector<int64_t> indices;
    for (int64_t lane = 0; lane < count; ++lane) indices.push_back(lane);
    Type outputType = isa<VectorType>(input.getType()) ? Type(cast<VectorType>(input.getType()).clone(outputElement)) : outputElement;
    return permute(loc, output, indices, outputType);
  }
  LogicalResult regionReduction(wk::ReduceRegionOp op) {
    auto dimensions = shape(op.getInputs()[0].getType());
    int64_t axis = op.getAxis(), extent = dimensions[axis];
    auto outputShape = shape(op.getResult(0).getType()), outputAxes = axes(op.getResult(0).getType());
    SmallVector<SmallVector<Value>> partials;
    SmallVector<Value> validParts;
    Value inputValid = valid(op.getInputs()[0], op.getLoc());
    for (Value value : op.getInputs().drop_front()) inputValid = arith::AndIOp::create(b, op.getLoc(), inputValid, valid(value, op.getLoc()));
    int64_t inner = volume(ArrayRef(dimensions).drop_front(axis + 1));
    for (int64_t k = 0; k < extent; ++k) {
      SmallVector<int64_t> indices;
      for (int64_t lane = 0; lane < volume(outputShape); ++lane)
        indices.push_back((lane / inner * extent + k) * inner + lane % inner);
      SmallVector<Value> tuple;
      for (auto [input, output] : llvm::zip(op.getInputs(), op.getResults()))
        tuple.push_back(permute(op.getLoc(), mapped(input), indices, physicalType(output.getType())));
      partials.push_back(std::move(tuple));
      validParts.push_back(permute(op.getLoc(), inputValid, indices, physicalType(maskType(op.getResult(0).getType()))));
    }
    while (partials.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      SmallVector<Value> nextValid;
      for (size_t i = 0; i < partials.size(); i += 2) {
        if (i + 1 == partials.size()) { next.push_back(std::move(partials[i])); nextValid.push_back(validParts[i]); }
        else {
          auto result = maskedCombine(op.getCombine(), partials[i], partials[i + 1], validParts[i], validParts[i + 1], op.getLoc());
          if (failed(result)) return failure();
          next.push_back(std::move(*result));
          nextValid.push_back(arith::OrIOp::create(b, op.getLoc(), validParts[i], validParts[i + 1]));
        }
      }
      partials = std::move(next);
      validParts = std::move(nextValid);
    }
    values.map(op.getResults(), partials.front());
    for (Value result : op.getResults()) masks[result] = validParts.front();
    return success();
  }
  LogicalResult scan(wk::ScanOp op) {
    auto dimensions = shape(op.getInputs()[0].getType()), ids = axes(op.getInputs()[0].getType());
    int64_t axis = op.getAxis(), extent = dimensions[axis];
    int64_t inner = volume(ArrayRef(dimensions).drop_front(axis + 1)), count = volume(dimensions);
    SmallVector<Value> partials;
    for (Value input : op.getInputs()) partials.push_back(mapped(input));
    Value originalValid = valid(op.getInputs()[0], op.getLoc());
    for (Value input : op.getInputs().drop_front()) originalValid = arith::AndIOp::create(b, op.getLoc(), originalValid, valid(input, op.getLoc()));
    Value partialValid = originalValid;
    for (int64_t distance = 1; distance < extent; distance *= 2) {
      SmallVector<int64_t> indices;
      SmallVector<APInt> active;
      for (int64_t lane = 0; lane < count; ++lane) {
        int64_t coordinate = lane / inner % extent;
        int64_t shifted = coordinate + (op.getReverse() ? distance : -distance);
        bool valid = shifted >= 0 && shifted < extent;
        indices.push_back(valid ? lane + (shifted - coordinate) * inner : lane);
        active.emplace_back(1, valid);
      }
      SmallVector<Value> left;
      for (Value partial : partials) left.push_back(permute(op.getLoc(), partial, indices, partial.getType()));
      Value leftValid = permute(op.getLoc(), partialValid, indices, partialValid.getType());
      auto combined = maskedCombine(op.getCombine(), left, partials, leftValid, partialValid, op.getLoc());
      if (failed(combined)) return failure();
      auto maskType = VectorType::get({count}, b.getI1Type());
      Value mask = arith::ConstantOp::create(b, op.getLoc(), maskType, DenseElementsAttr::get(maskType, active));
      for (size_t i = 0; i < partials.size(); ++i)
        partials[i] = arith::SelectOp::create(b, op.getLoc(), mask, (*combined)[i], partials[i]);
      partialValid = arith::SelectOp::create(b, op.getLoc(), mask,
          arith::OrIOp::create(b, op.getLoc(), leftValid, partialValid), partialValid);
    }
    values.map(op.getResults(), partials);
    for (Value result : op.getResults()) masks[result] = originalValid;
    return success();
  }
  LogicalResult grouped(Operation &op, ValueRange sources, int64_t group) {
    Location loc = op.getLoc();
    auto result = cast<wk::ValueType>(op.getResult(0).getType());
    auto outputShape = shape(result), outputAxes = axes(result);
    SmallVector<SmallVector<int64_t>> inputShapes;
    for (Value source : sources) inputShapes.push_back(shape(source.getType()));
    SmallVector<weft::analysis::LogicalDomain> inputDomains;
    for (auto [index, source] : llvm::enumerate(sources)) {
      auto type = cast<wk::ValueType>(source.getType());
      inputDomains.push_back({inputShapes[index], type.getAxisIds().asArrayRef()});
    }
    auto domain = weft::analysis::groupedAxes(inputDomains, {outputShape, outputAxes}, group);
    if (!domain) return op.emitError("grouped arithmetic has no complete logical axis/group relation");
    if (group == 2 && sources.size() == 1 && element(result).isInteger(32)) {
      auto input = dyn_cast<IntegerType>(element(sources.front().getType()));
      if (input && input.getWidth() <= (input.isUnsigned() ? 15u : 16u)) {
        Value data = mapped(sources.front()), active = valid(sources.front(), loc);
        Type onesType = cast<VectorType>(data.getType()).clone(b.getI16Type());
        Value ones = constant(loc, onesType, b.getI16IntegerAttr(1));
        auto selected = weft::selectX86IntegerPairs(module, b, loc, data, ones, active, active,
            input, IntegerType::get(b.getContext(), 16, IntegerType::Signed), 32, target.vectorBits);
        if (selected) {
          values.map(op.getResult(0), selected->value); masks[op.getResult(0)] = selected->valid;
          return success();
        }
      }
    }
    if ((group == 2 || group == 4) && sources.size() == 2 &&
        shape(sources[0].getType()).back() == outputShape.back() * group &&
        shape(sources[1].getType()).back() == outputShape.back() * group) {
      auto lhsElement = dyn_cast<IntegerType>(element(sources[0].getType()));
      auto rhsElement = dyn_cast<IntegerType>(element(sources[1].getType()));
      auto resultElement = dyn_cast<IntegerType>(element(result));
      if (lhsElement && rhsElement && resultElement) {
        auto pairShape = outputShape;
        pairShape.back() *= group;
        auto lhsType = wk::ValueType::get(b.getContext(), lhsElement,
            b.getDenseI64ArrayAttr(pairShape), result.getAxisIds());
        auto rhsType = wk::ValueType::get(b.getContext(), rhsElement,
            b.getDenseI64ArrayAttr(pairShape), result.getAxisIds());
        Value lhs = aligned(sources[0], lhsType, loc), rhs = aligned(sources[1], rhsType, loc);
        Value lhsValid = broadcast(loc, valid(sources[0], loc), maskType(sources[0].getType()), maskType(lhsType));
        Value rhsValid = broadcast(loc, valid(sources[1], loc), maskType(sources[1].getType()), maskType(rhsType));
        std::optional<weft::IntegerPairResult> selected;
        if (group == 4 && resultElement.getWidth() <= 32) {
          selected = weft::selectX86IntegerQuads(module, b, loc, lhs, rhs, lhsValid, rhsValid,
              lhsElement, rhsElement, target);
          // A wider modular sum realizes the same low bits. Restore the
          // author's boundary before any later widening or floating conversion.
          if (selected && resultElement.getWidth() < 32)
            selected->value = arith::TruncIOp::create(b, loc,
                cast<VectorType>(selected->value.getType()).clone(b.getIntegerType(resultElement.getWidth())),
                selected->value);
        }
        if (group == 2)
          selected = weft::selectX86IntegerPairs(module, b, loc, lhs, rhs, lhsValid, rhsValid,
              lhsElement, rhsElement, resultElement.getWidth(), target.vectorBits);
        if (selected) {
          values.map(op.getResult(0), selected->value);
          masks[op.getResult(0)] = selected->valid;
          return success();
        }
      }
    }
    if (group > 1048576 / volume(outputShape)) {
      auto diagnostic = op.emitError("x86 static grouped operation exceeds the 1048576 element-issue budget; use source tiles");
      diagnostic.getUnderlyingDiagnostic()->getMetadata().emplace_back(StringRef("weft.candidate.resource-limit"));
      return failure();
    }
    Type physical = physicalType(result), maskPhysical = physicalType(maskType(result));
    Value accumulator = zero(loc, physical), active = zero(loc, maskPhysical);
    for (int64_t member = 0; member < group; ++member) {
      SmallVector<Value> operands;
      Value memberActive = constant(loc, maskPhysical, b.getBoolAttr(true));
      for (auto [sourceNumber, source] : llvm::enumerate(sources)) {
        auto inputShape = shape(source.getType()), inputAxes = axes(source.getType());
        SmallVector<int64_t> indices;
        for (int64_t lane = 0; lane < volume(outputShape); ++lane) {
          int64_t index = 0;
          for (auto [position, axis] : llvm::enumerate(inputAxes)) {
            int64_t outputAxis = domain->inputToOutput[sourceNumber][position];
            int64_t coordinate = lane / volume(ArrayRef(outputShape).drop_front(outputAxis + 1)) % outputShape[outputAxis];
            if (position + 1 == inputAxes.size()) coordinate = coordinate * group + member;
            else if (inputShape[position] == 1) coordinate = 0;
            index = index * inputShape[position] + coordinate;
          }
          indices.push_back(index);
        }
        Type selectedType = wk::ValueType::get(b.getContext(), element(source.getType()),
            result.getShape(), result.getAxisIds());
        Value selected = permute(loc, mapped(source), indices, physicalType(selectedType));
        // Convert each operand before multiplying: the source explicitly owns
        // the product/accumulator width, including integer wrap boundaries.
        operands.push_back(castValue(loc, selected, selectedType, result));
        memberActive = arith::AndIOp::create(b, loc, memberActive,
            permute(loc, valid(source, loc), indices, maskPhysical));
      }
      for (Value &operand : operands)
        operand = arith::SelectOp::create(b, loc, memberActive, operand, zero(loc, physical));
      active = arith::OrIOp::create(b, loc, active, memberActive);
      bool floating = isa<FloatType>(scalar(result));
      if (operands.size() == 2 && floating) {
        accumulator = operation(loc, "math.fma", physical, {operands[0], operands[1], accumulator});
      } else {
        Value term = operands.size() == 2 ? operation(loc, "arith.muli", physical, operands) : operands[0];
        accumulator = operation(loc, floating ? "arith.addf" : "arith.addi", physical, {accumulator, term});
      }
    }
    values.map(op.getResult(0), accumulator);
    masks[op.getResult(0)] = active;
    return success();
  }
  LogicalResult lookup(wk::LookupOp op) {
    Location loc = op.getLoc();
    Type result = physicalType(op.getResult().getType());
    Type resultMask = physicalType(maskType(op.getResult().getType()));
    auto tableType = cast<VectorType>(mapped(op.getTable()).getType());
    Value indexValid = valid(op.getIndices(), loc);
    Value safeIndices = arith::SelectOp::create(b, loc, indexValid,
        mapped(op.getIndices()), zero(loc, mapped(op.getIndices()).getType()));
    if (Value selected = weft::x86::registerLookup(module, b, loc,
            mapped(op.getTable()), safeIndices, target.vectorBits)) {
      Value tableValid = weft::x86::registerLookup(module, b, loc,
          valid(op.getTable(), loc), safeIndices, target.vectorBits);
      assert(tableValid && "payload register table also admits predicate lookup");
      Value active = arith::AndIOp::create(b, loc, indexValid, tableValid);
      values.map(op.getResult(), arith::SelectOp::create(b, loc,
          active, selected, zero(loc, result)));
      masks[op.getResult()] = active;
      return success();
    }
    if (tableType.getNumElements() * tableType.getElementTypeBitWidth() > target.vectorBits) {
      // Dynamic extraction from a wide register value can spill and rebuild
      // the entire table for every selected lane. Keep the already-loaded SSA
      // snapshot in addressable local storage and issue indexed reads instead.
      auto snapshot = [&](Value value, Type storageElement) -> Value {
        auto vector = cast<VectorType>(value.getType());
        Type storedType = vector.clone(storageElement);
        if (vector != storedType)
          value = operation(loc, isa<FloatType>(vector.getElementType()) ? "arith.bitcast" : "arith.extui", storedType, value);
        // Keep the snapshot's allocation and effects in its actual scope.
        // LocalStorage and the pipeline passes own physical slot residency;
        // an entry-block LLVM pointer would hide per-iteration ownership.
        auto storageType = MemRefType::get({vector.getNumElements()}, storageElement);
        auto storage = memref::AllocOp::create(b, loc, storageType);
        storage.setAlignment(target.vectorBits / 8);
        auto flat = VectorType::get({vector.getNumElements()}, storageElement);
        if (value.getType() != flat) value = vector::ShapeCastOp::create(b, loc, flat, value);
        vector::StoreOp::create(b, loc, value, storage, ValueRange{indexConstant(loc, 0)});
        return storage;
      };
      Type elem = tableType.getElementType(), storageElement = elem;
      if (auto integer = dyn_cast<IntegerType>(elem)) {
        unsigned width = 8;
        while (width < integer.getWidth()) width *= 2;
        storageElement = b.getIntegerType(width);
      } else if (elem.isBF16()) storageElement = b.getI16Type();
      Value payload = snapshot(mapped(op.getTable()), storageElement);
      Value tableValid = valid(op.getTable(), loc);
      auto maskConstant = tableValid.getDefiningOp<arith::ConstantOp>();
      auto maskBits = maskConstant ? dyn_cast<DenseIntElementsAttr>(maskConstant.getValue()) : DenseIntElementsAttr{};
      bool fullyValid = maskBits && maskBits.isSplat() && maskBits.getSplatValue<APInt>().isAllOnes();
      Value validity = fullyValid ? Value() : snapshot(tableValid, b.getI8Type());
      Type domain = op.getResult().getType();
      auto withElement = [&](Type element) -> Type {
        if (auto integer = dyn_cast<IntegerType>(element); integer && integer.getWidth() > 1)
          element = b.getIntegerType(integer.getWidth(), false);
        if (auto value = dyn_cast<wk::ValueType>(domain))
          return wk::ValueType::get(b.getContext(), element, value.getShape(), value.getAxisIds());
        return element;
      };
      Value indices = castValue(loc, mapped(op.getIndices()), op.getIndices().getType(), withElement(b.getI64Type()));
      Value active = valid(op.getIndices(), loc);
      auto gather = [&](Value storage, Type element, Value mask) -> Value {
        auto shaped = dyn_cast<VectorType>(indices.getType());
        if (shaped) {
          auto vector = shaped.clone(element);
          return vector::GatherOp::create(b, loc, vector, storage, ValueRange{indexConstant(loc, 0)}, indices, mask, zero(loc, vector));
        }
        auto branch = scf::IfOp::create(b, loc, TypeRange{element}, mask, true);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(branch.thenBlock());
        Value offset = arith::IndexCastOp::create(b, loc, b.getIndexType(), indices);
        auto loaded = memref::LoadOp::create(b, loc, storage, ValueRange{offset});
        scf::YieldOp::create(b, loc, loaded.getResult());
        b.setInsertionPointToStart(branch.elseBlock());
        scf::YieldOp::create(b, loc, zero(loc, element));
        return branch.getResult(0);
      };
      if (!fullyValid) {
        Value bytes = gather(validity, b.getI8Type(), active);
        Value selectedValid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, bytes, zero(loc, bytes.getType()));
        active = arith::AndIOp::create(b, loc, active, selectedValid);
      }
      Value selected = gather(payload, storageElement, active);
      if (storageElement != elem)
        selected = operation(loc, isa<FloatType>(elem) ? "arith.bitcast" : "arith.trunci", result, selected);
      values.map(op.getResult(), selected);
      masks[op.getResult()] = active;
      return success();
    }
    Value output = zero(loc, result), outputMask = zero(loc, resultMask);
    Value indices = mapped(op.getIndices());
    for (int64_t lane = 0; lane < volume(shape(op.getResult().getType())); ++lane) {
      auto extract = [&](Value value) -> Value {
        return isa<VectorType>(value.getType()) ? Value(vector::ExtractOp::create(b, loc, value, ArrayRef<int64_t>{lane})) : value;
      };
      Value index = extract(indices), active = extract(indexValid);
      // Inactive logical lanes have no index-domain obligation. Select a safe
      // coordinate before extraction so their payload cannot introduce poison.
      index = arith::SelectOp::create(b, loc, active, index, zero(loc, index.getType()));
      index = castValue(loc, index, element(op.getIndices().getType()), b.getI64Type());
      Value selected = operation(loc, "llvm.extractelement", scalar(op.getResult().getType()), {mapped(op.getTable()), index});
      Value selectedValid = operation(loc, "llvm.extractelement", b.getI1Type(), {valid(op.getTable(), loc), index});
      active = arith::AndIOp::create(b, loc, active, selectedValid);
      selected = arith::SelectOp::create(b, loc, active, selected, zero(loc, selected.getType()));
      if (isa<VectorType>(result)) {
        output = vector::InsertOp::create(b, loc, selected, output, ArrayRef<int64_t>{lane});
        outputMask = vector::InsertOp::create(b, loc, active, outputMask, ArrayRef<int64_t>{lane});
      } else { output = selected; outputMask = active; }
    }
    values.map(op.getResult(), output);
    masks[op.getResult()] = outputMask;
    return success();
  }
  Value lookupRecordRuns(Location loc, Value base, Value offsets, Value active,
                         Value fill, Type domain, int64_t run) {
    auto payload = cast<VectorType>(fill.getType());
    int64_t elementBits = payload.getElementTypeBitWidth();
    int64_t wordBits = run * elementBits, rows = payload.getNumElements() / run;
    if (elementBits < 8 || elementBits > 16 || (wordBits != 32 && wordBits != 64) || rows < 4)
      return accessRuns(loc, base, offsets, active, fill, domain, true, run);
    // A lookup row is a single exact-width memory word. Gather those words
    // while the independent row axis is still vector-shaped; scalar loads
    // followed by inserts otherwise force every address through a GPR and
    // reconstruct the result with a long sequence of cross-lane shuffles.
    auto words = VectorType::get({rows}, b.getIntegerType(wordBits));
    auto predicates = VectorType::get({rows}, b.getI1Type());
    SmallVector<int64_t> first(rows);
    for (int64_t row = 0; row < rows; ++row) first[row] = row * run;
    Value rowOffsets = permute(loc, offsets, first,
        cast<VectorType>(offsets.getType()).cloneWith({rows}, scalar(offsets.getType())));
    Value all = constant(loc, predicates, b.getBoolAttr(true));
    Value any = zero(loc, predicates);
    for (int64_t member = 0; member < run; ++member) {
      SmallVector<int64_t> indices(first);
      for (int64_t &index : indices) index += member;
      Value mask = permute(loc, active, indices, predicates);
      all = arith::AndIOp::create(b, loc, all, mask);
      any = arith::OrIOp::create(b, loc, any, mask);
    }
    // Reading a word is valid only when the entire row is active. Empty rows
    // issue no access. A partial row keeps the original per-element mask; no
    // widened access is justified by a neighbouring active lane.
    Value uniform = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND,
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, all, any));
    auto choice = scf::IfOp::create(b, loc, TypeRange{payload}, uniform, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&choice.getThenRegion().front());
      auto pointer = LLVM::LLVMPointerType::get(b.getContext());
      Value addresses = LLVM::GEPOp::create(b, loc, VectorType::get({rows}, pointer),
          payload.getElementType(), base, ValueRange{rowOffsets});
      Value wordFill = vector::BitCastOp::create(b, loc, words, fill);
      Value loaded = LLVM::masked_gather::create(b, loc, words, addresses, all,
          ValueRange{wordFill}, b.getI32IntegerAttr(1));
      scf::YieldOp::create(b, loc, vector::BitCastOp::create(b, loc, payload, loaded).getResult());
      b.setInsertionPointToStart(&choice.getElseRegion().front());
      scf::YieldOp::create(b, loc, accessRuns(loc, base, offsets, active, fill, domain, true, run));
    }
    return choice.getResult(0);
  }
  Value lookupTransfer(Location loc, Value base, Value offsets, Value active,
                       Value fill, Type domain, Value logicalIndices,
                       bool linearStorage = true) {
    auto sizes = shape(domain);
    auto ids = axes(logicalIndices.getType());
    auto step = ids.empty() ? std::nullopt : modularAxisStep(logicalIndices, ids.back());
    int64_t run = sizes.empty() ? 1 : std::min<int64_t>(sizes.back(),
        target.vectorBits / std::max<unsigned>(8, scalar(domain).getIntOrFloatBitWidth()));
    if (step == std::optional<int64_t>(1) && run > 1 && sizes.back() % run == 0) {
      if (linearStorage && weft::x86::hasNonWrappingUnitStride(logicalIndices, ids.back()))
        return lookupRecordRuns(loc, base, offsets, active, fill, domain, run);
      return accessCheckedRuns(loc, base, offsets, active, fill, domain, true, run);
    }
    return access(loc, base, offsets, active, fill, domain, true, false);
  }
  // Only a sole lookup in the same read-only interval can consume the memory
  // snapshot directly. A staged/shared table or an intervening write keeps
  // the complete saved Value. This is demand-driven storage selection, not
  // permission to reload a snapshot after a write or in another source region.
  bool selectIndexedSnapshot(wk::AdmitOp admit) {
    Value result = admit.getResult();
    if (!result.hasOneUse() || packedType(result.getType())) return false;
    auto lookup = dyn_cast<wk::LookupOp>(*result.getUsers().begin());
    if (!lookup || lookup.getTable() != result || lookup->getBlock() != admit->getBlock() ||
        arrayValue(lookup.getResult().getType())) return false;
    auto type = dyn_cast<VectorType>(physicalType(result.getType()));
    if (type && type.getNumElements() * type.getElementTypeBitWidth() <= target.vectorBits)
      return false;
    for (Operation *next = admit->getNextNode(); next != lookup; next = next->getNextNode()) {
      if (!next) return false;
      auto effects = getEffectsRecursively(next);
      if (!effects || llvm::any_of(*effects, [](const auto &effect) {
            return !isa<MemoryEffects::Read>(effect.getEffect());
          })) return false;
    }
    indexedSnapshots[result] = memories.at(admit.getRegion());
    return true;
  }
  LogicalResult indexedSnapshotLookup(wk::LookupOp op, const Memory &view) {
    Location loc = op.getLoc();
    Type domain = op.getResult().getType();
    auto asDomain = [&](Type elementType) -> Type {
      if (auto shaped = dyn_cast<wk::ValueType>(domain))
        return wk::ValueType::get(b.getContext(), elementType, shaped.getShape(), shaped.getAxisIds());
      return elementType;
    };
    Type addressDomain = asDomain(b.getIntegerType(64, IntegerType::Unsigned));
    Type addressType = physicalType(addressDomain), predicateType = physicalMask(domain);
    auto splat = [&](Value value, Type type) -> Value {
      return isa<VectorType>(type) ? Value(vector::BroadcastOp::create(b, loc, type, value)) : value;
    };
    auto asAddress = [&](Value value) -> Value {
      Value integer = arith::IndexCastUIOp::create(b, loc, b.getI64Type(), value);
      return splat(integer, addressType);
    };
    Value active = arith::AndIOp::create(b, loc, valid(op.getIndices(), loc), splat(view.guard, predicateType));
    Value remaining = castValue(loc, mapped(op.getIndices()), op.getIndices().getType(), addressDomain);
    // A complete dense table has no holes in its logical snapshot. in_bounds
    // is the author's index precondition; only a partial View needs per-axis
    // validity reconstruction. GEP is not inbounds and the transfer is masked,
    // so inactive offsets need not be changed to zero (which destroys their
    // proven contiguous relation).
    bool linearStorage = true, complete = true;
    int64_t expectedStride = 1;
    for (int64_t axis = view.extents.size() - 1; axis >= 0; --axis) {
      APInt stride, extent, limit;
      bool constantExtent = matchPattern(view.extents[axis], m_ConstantInt(&extent));
      linearStorage &= matchPattern(view.strides[axis], m_ConstantInt(&stride)) &&
          stride.getSExtValue() == expectedStride;
      complete &= view.extents[axis] == view.limits[axis] ||
          (constantExtent && matchPattern(view.limits[axis], m_ConstantInt(&limit)) && extent == limit);
      if (!constantExtent || extent.isNegative() || extent.getActiveBits() >= 63 ||
          (extent.getZExtValue() && uint64_t(expectedStride) > INT64_MAX / extent.getZExtValue())) {
        linearStorage = false;
        break;
      }
      expectedStride *= extent.getZExtValue();
    }
    Value offset = asAddress(view.offset);
    if (linearStorage && complete) {
      offset = arith::AddIOp::create(b, loc, offset, remaining);
    } else for (int64_t axis = view.extents.size() - 1; axis >= 0; --axis) {
      Value coordinate = remaining;
      if (axis) {
        // Empty logical rows have no active index. Avoid even an inactive
        // divide-by-zero before the actual transfer mask is applied.
        Value extent = asAddress(view.extents[axis]);
        Value nonempty = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, extent, zero(loc, addressType));
        active = arith::AndIOp::create(b, loc, active, nonempty);
        Value divisor = arith::SelectOp::create(b, loc, nonempty, extent,
            constant(loc, addressType, b.getI64IntegerAttr(1)));
        coordinate = arith::RemUIOp::create(b, loc, remaining, divisor);
        remaining = arith::DivUIOp::create(b, loc, remaining, divisor);
      }
      active = arith::AndIOp::create(b, loc, active,
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, coordinate, asAddress(view.limits[axis])));
      offset = arith::AddIOp::create(b, loc, offset,
          arith::MulIOp::create(b, loc, coordinate, asAddress(view.strides[axis])));
    }
    auto branch = scf::IfOp::create(b, loc, TypeRange{physicalType(domain)}, view.storageMapped, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      scf::YieldOp::create(b, loc, storageAccess(loc, view, offset, active,
          zero(loc, physicalType(domain)), domain, true));
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      scf::YieldOp::create(b, loc, lookupTransfer(loc, view.base, offset, active,
          zero(loc, physicalType(domain)), domain, op.getIndices(), linearStorage && complete));
    }
    values.map(op.getResult(), branch.getResult(0));
    masks[op.getResult()] = active;
    return success();
  }
  bool integerMatrixGeometry(Operation &op) {
    return weft::x86::integerMatrixGeometry(op, target, shape(op.getOperand(0).getType()),
                                           shape(op.getOperand(1).getType()));
  }
  FailureOr<weft::x86::ContractionEngine> selectContractionEngine(Operation &op, bool arrayResult) {
    weft::x86::ContractionStorage storage{
        isa<MemRefType>(mapped(op.getOperand(0)).getType()),
        isa<MemRefType>(mapped(op.getOperand(1)).getType()),
        isa<VectorType>(mapped(op.getOperand(1)).getType())};
    auto selected = weft::x86::selectContractionEngine(op, target, storage, arrayResult,
        shape(op.getOperand(0).getType()), shape(op.getOperand(1).getType()), shape(op.getResult(0).getType()));
    if (succeeded(selected)) weft::x86::recordContractionEngine(module, op, *selected);
    return selected;
  }
  LogicalResult contraction(Operation &op) {
    auto selected = selectContractionEngine(op, false);
    if (failed(selected)) return failure();
    if (*selected == weft::x86::ContractionEngine::Vector) return simdContraction(op);
    Location loc = op.getLoc();
    Value result = op.getResult(0);
    Type payloadType = physicalType(result.getType()), predicateType = physicalType(maskType(result.getType()));
    // Integer products admit exact zero padding. Preserve the masks on the
    // source reads, zero inactive saved lanes, and use the same closed tile
    // contraction on partial domains. Expanding a second arithmetic program
    // for the masked case creates enormous otherwise-cold SIMD expressions.
    auto left = weft::x86::matrixOperandShape(op, 0, shape(op.getOperand(0).getType()));
    auto right = weft::x86::matrixOperandShape(op, 1, shape(op.getOperand(1).getType()));
    // Project each saved SSA operand into the selected matrix layout. The
    // logical Canonical axes and the original masked reads remain unchanged.
    auto matrixLayout = [&](unsigned side, Value value) -> Value {
      if (!weft::x86::matrixReductionFirst(op, side)) return value;
      auto dimensions = side == 0 ? left : right;
      SmallVector<int64_t> order;
      for (int64_t row = 0; row < dimensions[0]; ++row)
        for (int64_t term = 0; term < dimensions[1]; ++term)
          order.push_back(term * dimensions[0] + row);
      return permute(loc, value, order, value.getType());
    };
    auto rowMask = VectorType::get({left[1]}, b.getI1Type());
    SmallVector<Value> leftMasks, rightMasks;
    for (unsigned side = 0; side < 2; ++side) {
      auto &shape = side == 0 ? left : right;
      auto &rows = side == 0 ? leftMasks : rightMasks;
      for (int64_t row = 0; row < shape[0]; ++row) {
        SmallVector<int64_t> indices(shape[1]);
        std::iota(indices.begin(), indices.end(), row * shape[1]);
        rows.push_back(permute(loc, matrixLayout(side, valid(op.getOperand(side), loc)), indices, rowMask));
      }
    }
    SmallVector<Value> outputMasks;
    for (Value lhs : leftMasks) for (Value rhs : rightMasks) {
      Value pair = arith::AndIOp::create(b, loc, lhs, rhs);
      outputMasks.push_back(vector::ReductionOp::create(b, loc, vector::CombiningKind::OR, pair));
    }
    Value active = vector::FromElementsOp::create(b, loc, predicateType, outputMasks);
    SmallVector<Value> payloads;
    SmallVector<bool> unsignedInputs;
    int wideInput = -1;
    for (auto [position, input] : llvm::enumerate(op.getOperands())) {
      Value payload = mapped(input);
      payload = arith::SelectOp::create(b, loc, valid(input, loc), payload, zero(loc, payload.getType()));
      payload = matrixLayout(position, payload);
      auto type = cast<VectorType>(payload.getType());
      unsigned width = type.getElementTypeBitWidth();
      if (width > 8) wideInput = position;
      unsigned carrier = width > 8 ? 16 : 8;
      if (width != carrier)
        payload = operation(loc, unsignedType(input.getType()) || type.getElementTypeBitWidth() == 1 ?
            "arith.extui" : "arith.extsi", type.clone(b.getIntegerType(carrier)), payload);
      payloads.push_back(payload);
      unsignedInputs.push_back(unsignedType(input.getType()));
    }
    auto leftStorage = [&](Value payload) {
      SmallVector<Value> dimensions{indexConstant(loc, left[0]), indexConstant(loc, left[1])};
      auto storage = weft::x86::allocateArray(b, loc, b.getI8Type(), dimensions);
      vector::StoreOp::create(b, loc, payload, storage.data, indexConstant(loc, 0));
      return storage;
    };
    auto emit = [&](const weft::x86::ArrayStorage &lhs, Value rhs, bool lhsUnsigned, bool rhsUnsigned) {
      auto output = weft::x86::allocateArray(b, loc, b.getI32Type(), arrayExtents(result, loc));
      return weft::x86::materializeAMXIntegerFixed(module, b, loc, target, lhs, rhs, output,
          left[0], right[0], left[1], lhsUnsigned, rhsUnsigned, true);
    };
    Value value;
    if (wideInput < 0) {
      value = emit(leftStorage(payloads[0]), payloads[1], unsignedInputs[0], unsignedInputs[1]);
    } else {
      // One word operand is exactly low_u8 + 256 * high_i8/u8. Two
      // closed byte contractions therefore implement the same I32 modular
      // product, including signed minima and overflow. The longer reduction
      // domain amortizes the extra tile transfers; no author scale or loop
      // is moved across the original contraction boundary.
      Value word = payloads[wideInput];
      auto type = cast<VectorType>(word.getType()), bytes = type.clone(b.getI8Type());
      Value low = arith::TruncIOp::create(b, loc, bytes, word);
      Value shift = constant(loc, type, b.getI16IntegerAttr(8));
      Value high = operation(loc, unsignedInputs[wideInput] ? "arith.shrui" : "arith.shrsi", type, {word, shift});
      high = arith::TruncIOp::create(b, loc, bytes, high);
      Value lower, upper;
      if (wideInput == 0) {
        lower = emit(leftStorage(low), payloads[1], true, unsignedInputs[1]);
        upper = emit(leftStorage(high), payloads[1], unsignedInputs[0], unsignedInputs[1]);
      } else {
        auto input = leftStorage(payloads[0]);
        lower = emit(input, low, unsignedInputs[0], true);
        upper = emit(input, high, unsignedInputs[0], unsignedInputs[1]);
      }
      Value upperValue = arith::ShLIOp::create(b, loc, upper,
          constant(loc, payloadType, b.getI32IntegerAttr(8)));
      value = arith::AddIOp::create(b, loc, lower, upperValue);
    }
    values.map(result, value);
    masks[result] = active;
    return success();
  }
  LogicalResult simdContraction(Operation &op) {
    auto over = op.getAttrOfType<DenseI64ArrayAttr>("over");
    Type result = op.getResult(0).getType();
    auto outputShape = shape(result), outputAxes = axes(result);
    auto leftShape = shape(op.getOperand(0).getType()), leftAxes = axes(op.getOperand(0).getType());
    SmallVector<int64_t> reducedShape;
    for (int64_t id : over.asArrayRef()) {
      auto found = llvm::find(leftAxes, id);
      if (found == leftAxes.end()) return op.emitError("x86 contraction reduction axis is absent");
      reducedShape.push_back(leftShape[found - leftAxes.begin()]);
    }
    if (volume(reducedShape) > 1048576 / volume(outputShape)) {
      auto diagnostic = op.emitError("x86 static contraction exceeds the 1048576 element-issue budget; use source tiles");
      diagnostic.getUnderlyingDiagnostic()->getMetadata().emplace_back(StringRef("weft.candidate.resource-limit"));
      return failure();
    }
    Location loc = op.getLoc();
    Type physical = physicalType(result);
    Value accumulator = zero(loc, physical);
    Type maskPhysical = physicalType(maskType(result));
    Value active = zero(loc, maskPhysical);
    // Integer accumulation is modular in the declared result width. Keep the
    // contracted coordinates in SIMD lanes instead of extracting each term
    // into a dependent scalar multiply/add chain. Conversion still precedes
    // multiplication; no narrow product or saturating instruction is implied.
    // A reduction-contiguous operand pair supplies SIMD packets without the
    // strided free-axis permutations of the outer-product representation.
    // Compare packet arithmetic and the horizontal combine with those lane
    // selections. This selects a representation inside the closed contraction;
    // source tiles, scale trees and surrounding accumulators are unchanged.
    bool reductionContiguous = over.size() == 1;
    for (Value source : op.getOperands())
      reductionContiguous &= !axes(source.getType()).empty() && axes(source.getType()).back() == over.asArrayRef().front();
    int64_t lanes = target.vectorBits / scalar(result).getIntOrFloatBitWidth();
    int64_t outputs = volume(outputShape), terms = volume(reducedShape);
    int64_t packets = (terms + lanes - 1) / lanes;
    int64_t freePackets = (outputs + lanes - 1) / lanes;
    int64_t combine = llvm::Log2_64_Ceil(std::max(int64_t(1), std::min(lanes, terms)));
    int64_t reductionWork = outputs * (3 * packets + combine);
    int64_t outputWork = terms * (3 * freePackets + outputs);
    auto lhsInteger = dyn_cast<IntegerType>(element(op.getOperand(0).getType()));
    auto rhsInteger = dyn_cast<IntegerType>(element(op.getOperand(1).getType()));
    bool byteInputs = lhsInteger && rhsInteger && lhsInteger.getWidth() <= 8 && rhsInteger.getWidth() <= 8;
    bool unsignedBytes = byteInputs && lhsInteger.isUnsigned() && rhsInteger.isUnsigned() &&
                         lhsInteger.getWidth() == 8 && rhsInteger.getWidth() == 8;
    if (target.hasByteDot() && scalar(result).isInteger(32) && byteInputs && !unsignedBytes &&
        isa<VectorType>(physical) && terms % 4 == 0) {
      int64_t byteLanes = (target.hasFeature("avx512vnni") ? target.vectorBits : 256) / 8;
      int64_t outputPackets = (outputs + byteLanes / 4 - 1) / (byteLanes / 4);
      // An operand without a varying output axis supplies the same four bytes
      // to every output lane: one broadcast, rather than a two-input packet
      // permutation. Account for that actual reuse before comparing reductions.
      int64_t packetWork = 1;
      for (Value source : op.getOperands()) {
        auto dimensions = shape(source.getType()), ids = axes(source.getType());
        bool varyingOutput = false;
        for (auto [dimension, id] : llvm::zip(dimensions, ids))
          varyingOutput |= dimension > 1 && llvm::is_contained(outputAxes, id);
        packetWork += varyingOutput ? 2 : 1;
      }
      int64_t groupedWork = terms / 4 * outputPackets * packetWork;
      int64_t horizontalWork = outputs * ((terms + byteLanes - 1) / byteLanes +
                                          llvm::Log2_64_Ceil(std::max<int64_t>(1, terms / 4)));
      SmallVector<int64_t> deaths(outputPackets, 0);
      int64_t residentInputs = 0;
      for (Value source : op.getOperands()) {
        auto dimensions = shape(source.getType()), ids = axes(source.getType());
        int64_t bits = element(source.getType()).getIntOrFloatBitWidth();
        SmallVector<int64_t> last(llvm::divideCeil(volume(dimensions) * bits, int64_t(target.vectorBits)), -1);
        for (int64_t output = 0; output < outputs; ++output)
          for (int64_t term = 0; term < terms; ++term) {
            int64_t index = 0;
            for (auto [position, id] : llvm::enumerate(ids)) {
              auto free = llvm::find(outputAxes, id);
              int64_t coordinate;
              if (free != outputAxes.end()) {
                int64_t axis = free - outputAxes.begin();
                coordinate = output / volume(ArrayRef(outputShape).drop_front(axis + 1)) % outputShape[axis];
              } else {
                auto reduced = llvm::find(over.asArrayRef(), id);
                if (reduced == over.asArrayRef().end()) return op.emitError("unbound x86 contraction axis");
                int64_t axis = reduced - over.asArrayRef().begin();
                coordinate = term / volume(ArrayRef(reducedShape).drop_front(axis + 1)) % reducedShape[axis];
              }
              index = index * dimensions[position] + (dimensions[position] == 1 ? 0 : coordinate);
            }
            last[index * bits / target.vectorBits] = output / (byteLanes / 4);
          }
        bool shared = source.getParentBlock() != op.getBlock() ||
            llvm::any_of(source.getUses(), [&](OpOperand &use) { return use.getOwner() != &op; });
        for (int64_t packet : last) if (packet >= 0) {
          ++deaths[shared ? outputPackets - 1 : packet]; ++residentInputs;
        }
      }
      // Finish one native output packet at a time. Completed results remain
      // live, while the two operands, product and validity scratch belong only
      // to the current packet. LLVM still owns surrounding allocation/spills.
      // A wide saved input is present in both representations. Rejecting this
      // candidate solely for its size can select many horizontal reductions
      // with even more work. Charge spill/reload traffic at each retirement
      // frontier instead. Only the current native issue requires concurrent
      // registers; LLVM places saved inputs and completed output packets.
      int64_t spillWork = 0;
      for (int64_t packet = 0; packet < outputPackets; ++packet) {
        int64_t excess = std::max<int64_t>(0, residentInputs + packet + 1 + 4 - target.vectorRegisters());
        spillWork += 2 * excess;
        residentInputs -= deaths[packet];
      }
      // On equal work, retain independent output accumulators instead of
      // introducing a horizontal dependency tree for each output.
      if (groupedWork + spillWork <= horizontalWork) {
        for (int64_t outputFirst = 0; outputFirst < outputs; outputFirst += byteLanes / 4) {
          int64_t count = std::min(outputs - outputFirst, byteLanes / 4);
          auto packetType = VectorType::get({count}, b.getI32Type());
          auto packetMask = packetType.clone(b.getI1Type());
          Value partial = zero(loc, packetType), partialActive = zero(loc, packetMask);
          auto pairType = VectorType::get({count * 4}, b.getI1Type());
          for (int64_t first = 0; first < terms; first += 4) {
            SmallVector<Value> operands, predicates;
            for (Value source : op.getOperands()) {
              auto dimensions = shape(source.getType()), ids = axes(source.getType());
              SmallVector<int64_t> indices;
              for (int64_t output = outputFirst; output < outputFirst + count; ++output) {
                for (int64_t k = first; k < first + 4; ++k) {
                  int64_t index = 0;
                  for (auto [position, id] : llvm::enumerate(ids)) {
                    auto free = llvm::find(outputAxes, id);
                    int64_t coordinate;
                    if (free != outputAxes.end()) {
                      int64_t d = free - outputAxes.begin();
                      coordinate = output / volume(ArrayRef(outputShape).drop_front(d + 1)) % outputShape[d];
                    } else {
                      auto reduced = llvm::find(over.asArrayRef(), id);
                      if (reduced == over.asArrayRef().end()) return op.emitError("unbound x86 contraction axis");
                      int64_t d = reduced - over.asArrayRef().begin();
                      coordinate = k / volume(ArrayRef(reducedShape).drop_front(d + 1)) % reducedShape[d];
                    }
                    index = index * dimensions[position] + (dimensions[position] == 1 ? 0 : coordinate);
                  }
                  indices.push_back(index);
                }
              }
              operands.push_back(permute(loc, mapped(source), indices, pairType.clone(scalar(source.getType()))));
              predicates.push_back(permute(loc, valid(source, loc), indices, pairType));
            }
            auto group = weft::selectX86IntegerQuads(module, b, loc, operands[0], operands[1],
                predicates[0], predicates[1], lhsInteger, rhsInteger, target);
            assert(group && "the byte contraction candidate satisfies the local instruction contract");
            partial = arith::AddIOp::create(b, loc, partial, group->value);
            partialActive = arith::OrIOp::create(b, loc, partialActive, group->valid);
          }
          accumulator = vector::InsertStridedSliceOp::create(b, loc, partial, accumulator,
              ArrayRef<int64_t>{outputFirst}, ArrayRef<int64_t>{1});
          active = vector::InsertStridedSliceOp::create(b, loc, partialActive, active,
              ArrayRef<int64_t>{outputFirst}, ArrayRef<int64_t>{1});
        }
        values.map(op.getResult(0), accumulator); masks[op.getResult(0)] = active;
        return success();
      }
    }
    auto signedWord = [](IntegerType type) {
      return type && type.getWidth() <= (type.isUnsigned() ? 15u : 16u);
    };
    if (scalar(result).isInteger(32) && signedWord(lhsInteger) && signedWord(rhsInteger) &&
        isa<VectorType>(physical) && terms % 2 == 0 && terms <= 128 && outputs % lanes == 0) {
      // Retire each input packet after its last output consumer. Counting all
      // input snapshots alongside every completed output rejects schedules
      // whose rows are consumed in order and no longer overlap in liveness.
      SmallVector<int64_t> deaths(freePackets, 0);
      int64_t residentInputs = 0;
      for (Value source : op.getOperands()) {
        auto dimensions = shape(source.getType()), ids = axes(source.getType());
        int64_t bits = element(source.getType()).getIntOrFloatBitWidth();
        SmallVector<int64_t> last(llvm::divideCeil(volume(dimensions) * bits, int64_t(target.vectorBits)), -1);
        for (int64_t output = 0; output < outputs; ++output)
          for (int64_t term = 0; term < terms; ++term) {
            int64_t index = 0;
            for (auto [position, id] : llvm::enumerate(ids)) {
              auto free = llvm::find(outputAxes, id);
              int64_t coordinate;
              if (free != outputAxes.end()) {
                int64_t axis = free - outputAxes.begin();
                coordinate = output / volume(ArrayRef(outputShape).drop_front(axis + 1)) % outputShape[axis];
              } else {
                auto reduced = llvm::find(over.asArrayRef(), id);
                if (reduced == over.asArrayRef().end()) return op.emitError("unbound x86 contraction axis");
                int64_t axis = reduced - over.asArrayRef().begin();
                coordinate = term / volume(ArrayRef(reducedShape).drop_front(axis + 1)) % reducedShape[axis];
              }
              index = index * dimensions[position] + (dimensions[position] == 1 ? 0 : coordinate);
            }
            last[index * bits / target.vectorBits] = output / lanes;
          }
        bool shared = source.getParentBlock() != op.getBlock() ||
            llvm::any_of(source.getUses(), [&](OpOperand &use) { return use.getOwner() != &op; });
        for (int64_t packet : last) if (packet >= 0) {
          ++deaths[shared ? freePackets - 1 : packet]; ++residentInputs;
        }
      }
      int64_t spillWork = 0;
      for (int64_t packet = 0; packet < freePackets; ++packet) {
        int64_t excess = std::max<int64_t>(0, residentInputs + packet + 1 + 3 - target.vectorRegisters());
        spillWork += 2 * excess;
        residentInputs -= deaths[packet];
      }
      int64_t groupedWork = terms / 2 * freePackets * 5;
      int64_t horizontalWork = outputs * ((terms + lanes * 2 - 1) / (lanes * 2) +
                                          llvm::Log2_64_Ceil(std::max<int64_t>(1, terms)));
      // Finish one output packet before constructing the next, matching the
      // last-use frontier above. Scratch belongs to the current pair issue.
      // Completed packets need not all be resident at once. Their spill cost
      // participates in selection, as it does for byte groups; a large result
      // is not itself a reason to emit one horizontal reduction per element.
      if (groupedWork + spillWork < horizontalWork) {
        auto pairMask = VectorType::get({lanes * 2}, b.getI1Type());
        auto packetType = VectorType::get({lanes}, b.getI32Type());
        auto packetMask = packetType.clone(b.getI1Type());
        for (int64_t first = 0; first < outputs; first += lanes) {
          Value sum = zero(loc, packetType), live = zero(loc, packetMask);
          for (int64_t k = 0; k < terms; k += 2) {
            SmallVector<Value> inputs, predicates;
            for (Value source : op.getOperands()) {
              auto dimensions = shape(source.getType()), ids = axes(source.getType());
              SmallVector<int64_t> indices;
              for (int64_t output = first; output < first + lanes; ++output)
                for (int64_t term = k; term < k + 2; ++term) {
                  int64_t index = 0;
                  for (auto [position, id] : llvm::enumerate(ids)) {
                    auto free = llvm::find(outputAxes, id);
                    int64_t coordinate;
                    if (free != outputAxes.end()) {
                      int64_t d = free - outputAxes.begin();
                      coordinate = output / volume(ArrayRef(outputShape).drop_front(d + 1)) % outputShape[d];
                    } else {
                      auto reduced = llvm::find(over.asArrayRef(), id);
                      if (reduced == over.asArrayRef().end()) return op.emitError("unbound x86 contraction axis");
                      int64_t d = reduced - over.asArrayRef().begin();
                      coordinate = term / volume(ArrayRef(reducedShape).drop_front(d + 1)) % reducedShape[d];
                    }
                    index = index * dimensions[position] + (dimensions[position] == 1 ? 0 : coordinate);
                  }
                  indices.push_back(index);
                }
              inputs.push_back(permute(loc, mapped(source), indices, pairMask.clone(scalar(source.getType()))));
              predicates.push_back(permute(loc, valid(source, loc), indices, pairMask));
            }
            auto selected = weft::selectX86IntegerPairs(module, b, loc, inputs[0], inputs[1],
                predicates[0], predicates[1], lhsInteger, rhsInteger, 32, target.vectorBits);
            assert(selected && "the signed-word contraction satisfies the exact pair contract");
            sum = arith::AddIOp::create(b, loc, sum, selected->value);
            live = arith::OrIOp::create(b, loc, live, selected->valid);
          }
          accumulator = vector::InsertStridedSliceOp::create(b, loc, sum, accumulator,
              ArrayRef<int64_t>{first}, ArrayRef<int64_t>{1});
          active = vector::InsertStridedSliceOp::create(b, loc, live, active,
              ArrayRef<int64_t>{first}, ArrayRef<int64_t>{1});
        }
        values.map(op.getResult(0), accumulator); masks[op.getResult(0)] = active;
        return success();
      }
    }
    bool reduceFloating = isa<FloatType>(scalar(result)) && reductionContiguous && reductionWork < outputWork;
    if ((isa<IntegerType>(scalar(result)) || reduceFloating) && terms <= 4096) {
      auto packetShape = b.getDenseI64ArrayAttr({volume(reducedShape)});
      auto packetAxes = b.getDenseI64ArrayAttr({over.asArrayRef().front()});
      Type packet = wk::ValueType::get(b.getContext(), element(result), packetShape, packetAxes);
      Type packetType = physicalType(packet);
      Type packetMask = cast<VectorType>(packetType).clone(b.getI1Type());
      for (int64_t lane = 0; lane < volume(outputShape); ++lane) {
        SmallVector<Value> inputs;
        Value pairActive = constant(loc, packetMask, b.getBoolAttr(true));
        for (Value source : op.getOperands()) {
          auto dimensions = shape(source.getType()), ids = axes(source.getType());
          SmallVector<int64_t> indices;
          for (int64_t k = 0; k < volume(reducedShape); ++k) {
            int64_t index = 0;
            for (auto [position, id] : llvm::enumerate(ids)) {
              auto free = llvm::find(outputAxes, id);
              int64_t coordinate;
              if (free != outputAxes.end()) {
                int64_t d = free - outputAxes.begin();
                coordinate = lane / volume(ArrayRef(outputShape).drop_front(d + 1)) % outputShape[d];
              } else {
                auto reduced = llvm::find(over.asArrayRef(), id);
                if (reduced == over.asArrayRef().end()) return op.emitError("unbound x86 contraction axis");
                int64_t d = reduced - over.asArrayRef().begin();
                coordinate = k / volume(ArrayRef(reducedShape).drop_front(d + 1)) % reducedShape[d];
              }
              index = index * dimensions[position] + (dimensions[position] == 1 ? 0 : coordinate);
            }
            indices.push_back(index);
          }
          Type selectedType = wk::ValueType::get(b.getContext(), element(source.getType()), packetShape, packetAxes);
          Value selected = permute(loc, mapped(source), indices, physicalType(selectedType));
          inputs.push_back(castValue(loc, selected, selectedType, packet));
          pairActive = arith::AndIOp::create(b, loc, pairActive,
              permute(loc, valid(source, loc), indices, packetMask));
        }
        Value product = operation(loc, isa<FloatType>(scalar(result)) ? "arith.mulf" : "arith.muli", packetType, inputs);
        product = arith::SelectOp::create(b, loc, pairActive, product, zero(loc, packetType));
        Value sum = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, product);
        Value live = vector::ReductionOp::create(b, loc, vector::CombiningKind::OR, pairActive);
        if (isa<VectorType>(physical)) {
          accumulator = vector::InsertOp::create(b, loc, sum, accumulator, ArrayRef<int64_t>{lane});
          active = vector::InsertOp::create(b, loc, live, active, ArrayRef<int64_t>{lane});
        } else { accumulator = sum; active = live; }
      }
      values.map(op.getResult(0), accumulator);
      masks[op.getResult(0)] = active;
      return success();
    }
    for (int64_t k = 0; k < volume(reducedShape); ++k) {
      SmallVector<Value> inputs;
      Value pairActive = constant(loc, maskPhysical, b.getBoolAttr(true));
      for (Value source : op.getOperands()) {
        auto dimensions = shape(source.getType()), ids = axes(source.getType());
        SmallVector<int64_t> indices;
        for (int64_t lane = 0; lane < volume(outputShape); ++lane) {
          int64_t index = 0;
          for (auto [position, id] : llvm::enumerate(ids)) {
            auto free = llvm::find(outputAxes, id);
            int64_t coordinate;
            if (free != outputAxes.end()) {
              int64_t d = free - outputAxes.begin();
              coordinate = lane / volume(ArrayRef(outputShape).drop_front(d + 1)) % outputShape[d];
            } else {
              auto reduced = llvm::find(over.asArrayRef(), id);
              if (reduced == over.asArrayRef().end()) return op.emitError("unbound x86 contraction axis");
              int64_t d = reduced - over.asArrayRef().begin();
              coordinate = k / volume(ArrayRef(reducedShape).drop_front(d + 1)) % reducedShape[d];
            }
            index = index * dimensions[position] + (dimensions[position] == 1 ? 0 : coordinate);
          }
          indices.push_back(index);
        }
        Type expanded = isa<wk::ValueType>(result) ? Type(wk::ValueType::get(b.getContext(), element(source.getType()),
            cast<wk::ValueType>(result).getShape(), cast<wk::ValueType>(result).getAxisIds())) : element(source.getType());
        Value value = permute(loc, mapped(source), indices, physicalType(expanded));
        inputs.push_back(castValue(loc, value, expanded, result));
        pairActive = arith::AndIOp::create(b, loc, pairActive, permute(loc, valid(source, loc), indices, maskPhysical));
      }
      for (Value &input : inputs) input = arith::SelectOp::create(b, loc, pairActive, input, zero(loc, physical));
      active = arith::OrIOp::create(b, loc, active, pairActive);
      if (isa<FloatType>(scalar(result)))
        accumulator = operation(loc, "math.fma", physical, {inputs[0], inputs[1], accumulator});
      else {
        Value product = operation(loc, "arith.muli", physical, inputs);
        accumulator = operation(loc, "arith.addi", physical, {accumulator, product});
      }
    }
    values.map(op.getResult(0), accumulator);
    masks[op.getResult(0)] = active;
    return success();
  }
  LogicalResult level(wk::LevelOp op) {
    if (cfgFunction) return cfgLevel(op);
    auto domain = op.getDomain().getDefiningOp<wk::DomainOp>();
    if (!domain) return op.emitError("x86 Level requires its verified domain definition");
    Location loc = op.getLoc();
    auto type = domain.getResult().getType();
    Value begin = arith::ConstantIndexOp::create(b, loc, 0);
    Value origin = begin, total = mapped(domain.getExtent());
    // Refine the nearest ancestor of the same logical axis, even when levels
    // for other axes intervene. The parent's tail, not its nominal partition,
    // bounds the child's instances.
    for (const Point &ancestor : llvm::reverse(activePoints)) {
      if (ancestor.domain.getAxisId() != type.getAxisId()) continue;
      origin = ancestor.base;
      total = ancestor.active;
      break;
    }
    Value partition = mapped(domain.getPartition());
    SmallVector<Value> carried;
    for (Value value : op.getCarried()) {
      if (carryTypes(value.getType()).empty())
        return unsupported(op, "unsupported x86 Level carry type: ") << value.getType();
      llvm::append_range(carried, carryValues(value, loc));
    }
    auto loop = scf::ForOp::create(b, loc, begin, total, partition, carried);
    loop->setAttr("weft.x86.level", b.getDictionaryAttr({
        b.getNamedAttr("domain", b.getI64IntegerAttr(type.getDomainId())),
        b.getNamedAttr("axis", b.getI64IntegerAttr(type.getAxisId())),
        b.getNamedAttr("parent", b.getI64IntegerAttr(type.getParentDomainId())),
        b.getNamedAttr("relation", b.getStringAttr(type.getRelation())),
        b.getNamedAttr("tail", b.getStringAttr(type.getTail()))}));
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value base = arith::AddIOp::create(b, loc, origin, loop.getInductionVar());
      Value remaining = arith::SubIOp::create(b, loc, total, loop.getInductionVar());
      Value active = arith::MinUIOp::create(b, loc, remaining, partition);
      Point point{type, base, active, partition};
      activePoints.push_back(point);
      auto pop = llvm::scope_exit([&] { activePoints.pop_back(); });
      SmallVector<Value> births;
      // The two birth regions are independent lexical scopes. They execute in
      // order once per instance and remain inside the physical loop. Their
      // typed SSA results feed the body, never the next instance's initializers.
      for (auto [region, name] : {std::make_pair(&op.getStateBirths(), "state"),
                                  std::make_pair(&op.getStagedBirths(), "stage")}) {
        Block &source = region->front();
        points[source.getArgument(0)] = point;
        auto yield = cast<wk::BirthsYieldOp>(source.getTerminator());
        SmallVector<Type> results;
        for (Value value : yield.getValues()) {
          auto result = carryTypes(value.getType());
          if (result.empty()) return unsupported(op, "unsupported x86 Level birth type: ") << value.getType();
          llvm::append_range(results, result);
        }
        auto scope = scf::ExecuteRegionOp::create(b, loc, results);
        scope->setAttr("weft.x86.birth", b.getStringAttr(name));
        scope->setAttr("weft.x86.owner", b.getI64IntegerAttr(type.getDomainId()));
        scope.getRegion().push_back(new Block);
        {
          OpBuilder::InsertionGuard birthGuard(b);
          b.setInsertionPointToStart(&scope.getRegion().front());
          for (Operation &operation : source.without_terminator())
            if (failed(lower(operation))) return failure();
          SmallVector<Value> output;
          for (Value value : yield.getValues()) llvm::append_range(output, carryValues(value, loc));
          scf::YieldOp::create(b, loc, output);
        }
        llvm::append_range(births, scope.getResults());
      }
      Block &body = op.getBody().front();
      points[body.getArgument(0)] = point;
      unsigned cursor = 0;
      for (Value value : body.getArguments().slice(1, op.getCarried().size())) bindCarry(value, loop.getRegionIterArgs(), cursor);
      cursor = 0;
      for (Value value : body.getArguments().drop_front(1 + op.getCarried().size())) bindCarry(value, births, cursor);
      for (Operation &operation : body.without_terminator())
        if (failed(lower(operation))) return failure();
      SmallVector<Value> output;
      for (Value value : cast<wk::HandoffOp>(body.getTerminator()).getValues()) llvm::append_range(output, carryValues(value, loc));
      if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back()))
        loop.getBody()->back().erase();
      b.setInsertionPointToEnd(loop.getBody());
      scf::YieldOp::create(b, loc, output);
    }
    unsigned cursor = 0;
    for (Value value : op.getResults()) bindCarry(value, loop.getResults(), cursor);
    return success();
  }
  LogicalResult projection(Operation &op, Value source, ValueRange indices,
                           ArrayAttr selectors, Value update = {}) {
    Location loc = op.getLoc();
    Type projected = update ? update.getType() : op.getResult(0).getType();
    auto dimensions = shape(source.getType()), sourceAxes = axes(source.getType());
    auto outputShape = shape(projected), outputAxes = axes(projected);
    Type outputType = physicalType(op.getResult(0).getType());
    auto sourceOrigins = origins(source, loc);
    SmallVector<Value> outputOrigins;
    for (int64_t axis : outputAxes) {
      auto found = llvm::find(sourceAxes, axis);
      outputOrigins.push_back(found == sourceAxes.end() ? indexConstant(loc, 0) : sourceOrigins[found - sourceAxes.begin()]);
    }
    int64_t recordElements = 1, recordBytes = 1;
    if (auto encoding = packedType(source.getType())) {
      auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
      if (failed(layout)) return unsupported(op, "x86 encoded projection requires a closed record layout");
      recordElements = layout->elements(); recordBytes = layout->bytes();
      dimensions.back() = dimensions.back() / recordElements * recordBytes;
      outputShape.back() = outputShape.back() / recordElements * recordBytes;
    }
    SmallVector<Value> coordinates, domainLimits, indexMasks;
    SmallVector<bool> boundedIndices;
    unsigned cursor = 0;
    for (size_t axis = 0; axis < dimensions.size(); ++axis) {
      StringRef selector = axis < selectors.size() ? cast<StringAttr>(selectors[axis]).getValue() : "all";
      domainLimits.push_back({}); indexMasks.push_back({}); boundedIndices.push_back(false);
      if (selector == "all") { coordinates.push_back({}); continue; }
      Value index = indices[cursor++];
      if (selector == "domain" || selector == "group_index") {
        auto found = points.find(index);
        if (found == points.end()) return op.emitError("x86 local projection lost its Level point");
        Point point = found->second;
        Value coordinate = selector == "domain" ? point.base : Value(arith::DivUIOp::create(b, loc, point.base, point.partition));
        Value offset = arith::SubIOp::create(b, loc, coordinate, sourceOrigins[axis]);
        Value extent = point.active;
        if (packedType(source.getType()) && axis + 1 == dimensions.size()) {
          offset = arith::MulIOp::create(b, loc, arith::DivUIOp::create(b, loc, offset, indexConstant(loc, recordElements)), indexConstant(loc, recordBytes));
          extent = arith::MulIOp::create(b, loc, arith::DivUIOp::create(b, loc, extent, indexConstant(loc, recordElements)), indexConstant(loc, recordBytes));
        }
        coordinates.push_back(offset);
        if (selector == "domain") {
          domainLimits.back() = extent;
          auto outputAxis = llvm::find(outputAxes, sourceAxes[axis]);
          if (outputAxis == outputAxes.end()) return op.emitError("domain projection must retain its logical axis");
          outputOrigins[outputAxis - outputAxes.begin()] = coordinate;
        }
      } else if (selector == "index" || selector == "gather") {
        coordinates.push_back(selector == "gather" ? aligned(index, projected, loc) : mapped(index));
        indexMasks.back() = selector == "gather" ? validAs(index, projected, loc) : valid(index, loc);
        boundedIndices.back() = !packedType(source.getType()) &&
            weft::x86::hasUnsignedBound(index, dimensions[axis]);
      } else
        return unsupported(op, "unsupported x86 local Value projection: ") << selector;
    }
    Value output = update ? mapped(source) : zero(loc, outputType);
    Type outputMaskType = physicalMask(op.getResult(0).getType());
    Value outputMask = update ? valid(source, loc) : zero(loc, outputMaskType);
    for (int64_t lane = 0; lane < volume(outputShape); ++lane) {
      Value linear = arith::ConstantIndexOp::create(b, loc, 0);
      Value selectedActive = constant(loc, b.getI1Type(), b.getBoolAttr(true));
      for (size_t axis = 0; axis < dimensions.size(); ++axis) {
        Value coordinate = coordinates[axis];
        if (!coordinate || domainLimits[axis]) {
          auto found = llvm::find(outputAxes, sourceAxes[axis]);
          if (found == outputAxes.end()) return op.emitError("x86 projection lost an unselected logical axis");
          unsigned position = found - outputAxes.begin();
          int64_t c = lane / volume(ArrayRef(outputShape).drop_front(position + 1)) % outputShape[position];
          Value relative = indexConstant(loc, c);
          coordinate = coordinate ? Value(arith::AddIOp::create(b, loc, coordinate, relative)) : relative;
          if (domainLimits[axis]) selectedActive = arith::AndIOp::create(b, loc, selectedActive,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, relative, domainLimits[axis]));
        } else {
          if (isa<VectorType>(coordinate.getType()))
            coordinate = vector::ExtractOp::create(b, loc, coordinate, ArrayRef<int64_t>{lane});
          // Gather indices are unsigned; scalar selectors already have index
          // type. Conversion preserves high bits instead of sign-extending a
          // narrow unsigned coordinate.
          if (!coordinate.getType().isIndex())
            coordinate = arith::IndexCastUIOp::create(b, loc, b.getIndexType(), coordinate);
        }
        Value width = arith::ConstantIndexOp::create(b, loc, dimensions[axis]);
        if (!boundedIndices[axis])
          selectedActive = arith::AndIOp::create(b, loc, selectedActive,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, coordinate, width));
        if (Value mask = indexMasks[axis]) {
          if (isa<VectorType>(mask.getType())) mask = vector::ExtractOp::create(b, loc, mask, ArrayRef<int64_t>{lane});
          selectedActive = arith::AndIOp::create(b, loc, selectedActive, mask);
        }
        linear = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, linear, width), coordinate);
      }
      linear = arith::SelectOp::create(b, loc, selectedActive, linear, indexConstant(loc, 0));
      Value address = arith::IndexCastOp::create(b, loc, b.getI64Type(), linear);
      Type selectedType = packedType(source.getType()) ? Type(b.getI8Type()) : scalar(source.getType());
      if (update) {
        Value selected = mapped(update);
        Value selectedMask = valid(update, loc);
        if (isa<VectorType>(selected.getType()))
          selected = vector::ExtractOp::create(b, loc, selected, ArrayRef<int64_t>{lane});
        if (isa<VectorType>(selectedMask.getType()))
          selectedMask = vector::ExtractOp::create(b, loc, selectedMask, ArrayRef<int64_t>{lane});
        selected = arith::SelectOp::create(b, loc, selectedActive, selected,
            operation(loc, "llvm.extractelement", selectedType, {output, address}));
        selectedMask = arith::SelectOp::create(b, loc, selectedActive, selectedMask,
            operation(loc, "llvm.extractelement", b.getI1Type(), {outputMask, address}));
        output = operation(loc, "llvm.insertelement", outputType, {output, selected, address});
        outputMask = operation(loc, "llvm.insertelement", outputMaskType, {outputMask, selectedMask, address});
      } else {
        Value selected = operation(loc, "llvm.extractelement", selectedType, {mapped(source), address});
        Value selectedMask = arith::AndIOp::create(b, loc, selectedActive,
            operation(loc, "llvm.extractelement", b.getI1Type(), {valid(source, loc), address}));
        if (isa<VectorType>(outputType)) {
          Value position = arith::ConstantIntOp::create(b, loc, lane, 64);
          output = operation(loc, "llvm.insertelement", outputType, {output, selected, position});
          outputMask = operation(loc, "llvm.insertelement", outputMaskType, {outputMask, selectedMask, position});
        } else { output = selected; outputMask = selectedMask; }
      }
    }
    values.map(op.getResult(0), output);
    masks[op.getResult(0)] = outputMask;
    valueOrigins[op.getResult(0)] = update ? sourceOrigins : outputOrigins;
    return success();
  }
  LogicalResult block(Block &source) {
    for (Operation &op : source) {
      if (!b.getInsertionBlock()) break;
      if (failed(lower(op))) return failure();
    }
    return success();
  }
  SmallVector<Value> flatten(ValueRange input, Location loc) {
    SmallVector<Value> result;
    for (Value value : input) llvm::append_range(result, carryValues(value, loc));
    return result;
  }
  FailureOr<SmallVector<Type>> flattenTypes(TypeRange types, Operation *owner) {
    SmallVector<Type> result;
    for (Type type : types) {
      auto fields = carryTypes(type);
      if (fields.empty()) return unsupported(owner, "unsupported x86 native helper/control value: ") << type;
      llvm::append_range(result, fields);
    }
    return result;
  }
  Block *cfgBlock(Location loc, TypeRange types = {}) {
    auto *result = new Block;
    cfgFunction.getBody().push_back(result);
    for (Type type : types) result->addArgument(type, loc);
    return result;
  }
  void bindRange(ValueRange source, ValueRange physical) {
    unsigned cursor = 0;
    for (Value value : source) bindCarry(value, physical, cursor);
  }
  LogicalResult cfgSequence(Block &source) {
    for (Operation &op : source) {
      if (!b.getInsertionBlock()) break;
      if (isa<scf::YieldOp, scf::ConditionOp, wk::BirthsYieldOp, wk::HandoffOp>(op)) break;
      if (failed(lower(op))) return failure();
    }
    return success();
  }
  LogicalResult cfgIf(scf::IfOp op) {
    auto types = flattenTypes(op.getResultTypes(), op);
    if (failed(types)) return failure();
    Location loc = op.getLoc();
    Block *yes = cfgBlock(loc), *no = cfgBlock(loc), *join = cfgBlock(loc, *types);
    cf::CondBranchOp::create(b, loc, mapped(op.getCondition()), yes, ValueRange{}, no, ValueRange{});
    for (auto [region, destination] : {std::make_pair(&op.getThenRegion(), yes),
                                      std::make_pair(&op.getElseRegion(), no)}) {
      b.setInsertionPointToEnd(destination);
      SmallVector<Value> results;
      if (!region->empty()) {
        Block &body = region->front();
        if (failed(cfgSequence(body))) return failure();
        if (b.getInsertionBlock()) results = flatten(body.getTerminator()->getOperands(), loc);
      }
      if (b.getInsertionBlock()) cf::BranchOp::create(b, loc, join, results);
    }
    if (join->hasNoPredecessors()) {
      join->erase();
      b.clearInsertionPoint();
    } else {
      b.setInsertionPointToEnd(join);
      bindRange(op.getResults(), join->getArguments());
    }
    return success();
  }
  LogicalResult cfgFor(scf::ForOp op) {
    auto types = flattenTypes(op.getResultTypes(), op);
    if (failed(types)) return failure();
    Location loc = op.getLoc();
    Value begin = mapped(op.getLowerBound()), end = mapped(op.getUpperBound()), step = mapped(op.getStep());
    SmallVector<Type> headerTypes{begin.getType()};
    llvm::append_range(headerTypes, *types);
    Block *header = cfgBlock(loc, headerTypes), *body = cfgBlock(loc), *exit = cfgBlock(loc, *types);
    SmallVector<Value> initial{begin};
    llvm::append_range(initial, flatten(op.getInitArgs(), loc));
    cf::BranchOp::create(b, loc, header, initial);
    b.setInsertionPointToEnd(header);
    Value induction = header->getArgument(0);
    Value active = arith::CmpIOp::create(b, loc,
        op.getUnsignedCmp() ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt, induction, end);
    cf::CondBranchOp::create(b, loc, active, body, ValueRange{}, exit, header->getArguments().drop_front());
    b.setInsertionPointToEnd(body);
    values.map(op.getInductionVar(), induction);
    bindRange(op.getRegionIterArgs(), header->getArguments().drop_front());
    if (failed(cfgSequence(*op.getBody()))) return failure();
    if (b.getInsertionBlock()) {
      SmallVector<Value> next{arith::AddIOp::create(b, loc, induction, step)};
      llvm::append_range(next, flatten(op.getBody()->getTerminator()->getOperands(), loc));
      cf::BranchOp::create(b, loc, header, next);
    }
    b.setInsertionPointToEnd(exit);
    bindRange(op.getResults(), exit->getArguments());
    return success();
  }
  LogicalResult cfgWhile(scf::WhileOp op) {
    auto beforeTypes = flattenTypes(op.getInits().getTypes(), op);
    auto resultTypes = flattenTypes(op.getResultTypes(), op);
    if (failed(beforeTypes) || failed(resultTypes)) return failure();
    Location loc = op.getLoc();
    Block *before = cfgBlock(loc, *beforeTypes), *after = cfgBlock(loc, *resultTypes), *exit = cfgBlock(loc, *resultTypes);
    cf::BranchOp::create(b, loc, before, flatten(op.getInits(), loc));
    b.setInsertionPointToEnd(before);
    bindRange(op.getBeforeArguments(), before->getArguments());
    if (failed(cfgSequence(op.getBefore().front()))) return failure();
    if (!b.getInsertionBlock()) {
      after->erase(); exit->erase();
      return success();
    }
    auto condition = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    auto forwarded = flatten(condition.getArgs(), loc);
    cf::CondBranchOp::create(b, loc, mapped(condition.getCondition()), after, forwarded, exit, forwarded);
    b.setInsertionPointToEnd(after);
    bindRange(op.getAfterArguments(), after->getArguments());
    if (failed(cfgSequence(op.getAfter().front()))) return failure();
    if (b.getInsertionBlock())
      cf::BranchOp::create(b, loc, before, flatten(op.getAfter().front().getTerminator()->getOperands(), loc));
    b.setInsertionPointToEnd(exit);
    bindRange(op.getResults(), exit->getArguments());
    return success();
  }
  LogicalResult cfgLevel(wk::LevelOp op) {
    auto domain = op.getDomain().getDefiningOp<wk::DomainOp>();
    if (!domain) return op.emitError("x86 Level requires its verified domain definition");
    auto types = flattenTypes(op.getResultTypes(), op);
    if (failed(types)) return failure();
    Location loc = op.getLoc();
    auto type = domain.getResult().getType();
    Value begin = indexConstant(loc, 0), origin = begin, total = mapped(domain.getExtent());
    for (const Point &ancestor : llvm::reverse(activePoints)) {
      if (ancestor.domain.getAxisId() != type.getAxisId()) continue;
      origin = ancestor.base; total = ancestor.active; break;
    }
    Value partition = mapped(domain.getPartition());
    SmallVector<Type> headerTypes{b.getIndexType()};
    llvm::append_range(headerTypes, *types);
    Block *header = cfgBlock(loc, headerTypes), *body = cfgBlock(loc), *exit = cfgBlock(loc, *types);
    SmallVector<Value> initial{begin};
    llvm::append_range(initial, flatten(op.getCarried(), loc));
    cf::BranchOp::create(b, loc, header, initial);
    b.setInsertionPointToEnd(header);
    Value induction = header->getArgument(0);
    Value active = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, induction, total);
    auto branch = cf::CondBranchOp::create(b, loc, active, body, ValueRange{}, exit, header->getArguments().drop_front());
    branch->setAttr("weft.x86.level", b.getDictionaryAttr({
        b.getNamedAttr("domain", b.getI64IntegerAttr(type.getDomainId())),
        b.getNamedAttr("axis", b.getI64IntegerAttr(type.getAxisId())),
        b.getNamedAttr("parent", b.getI64IntegerAttr(type.getParentDomainId())),
        b.getNamedAttr("relation", b.getStringAttr(type.getRelation())),
        b.getNamedAttr("tail", b.getStringAttr(type.getTail()))}));
    b.setInsertionPointToEnd(body);
    Point point{type, arith::AddIOp::create(b, loc, origin, induction),
        arith::MinUIOp::create(b, loc, arith::SubIOp::create(b, loc, total, induction), partition), partition};
    activePoints.push_back(point);
    auto pop = llvm::scope_exit([&] { activePoints.pop_back(); });
    SmallVector<Value> births;
    for (auto [region, name] : {std::make_pair(&op.getStateBirths(), "state"),
                                std::make_pair(&op.getStagedBirths(), "stage")}) {
      Block &source = region->front();
      points[source.getArgument(0)] = point;
      if (failed(cfgSequence(source))) return failure();
      if (!b.getInsertionBlock()) return op.emitError("Level birth exited its enclosing helper");
      auto output = flatten(source.getTerminator()->getOperands(), loc);
      SmallVector<Type> fields;
      for (Value value : output) fields.push_back(value.getType());
      Block *next = cfgBlock(loc, fields);
      auto boundary = cf::BranchOp::create(b, loc, next, output);
      boundary->setAttr("weft.x86.birth", b.getStringAttr(name));
      boundary->setAttr("weft.x86.owner", b.getI64IntegerAttr(type.getDomainId()));
      b.setInsertionPointToEnd(next);
      llvm::append_range(births, next->getArguments());
    }
    Block &source = op.getBody().front();
    points[source.getArgument(0)] = point;
    bindRange(source.getArguments().slice(1, op.getCarried().size()), header->getArguments().drop_front());
    bindRange(source.getArguments().drop_front(1 + op.getCarried().size()), births);
    if (failed(cfgSequence(source))) return failure();
    if (b.getInsertionBlock()) {
      SmallVector<Value> next{arith::AddIOp::create(b, loc, induction, partition)};
      llvm::append_range(next, flatten(source.getTerminator()->getOperands(), loc));
      cf::BranchOp::create(b, loc, header, next);
    }
    b.setInsertionPointToEnd(exit);
    bindRange(op.getResults(), exit->getArguments());
    return success();
  }
  LogicalResult helper(wk::HelperOp op) {
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(op.getBody(), captures);
    // Domain identity stays in the source IR. The x86 loop consumes extent and
    // partition; multiplicity is derived by that loop, not a native argument.
    // Ancestor point coordinates are captured separately below.
    for (unsigned i = 0; i < captures.size(); ++i)
      if (isa<wk::DomainType>(captures[i].getType())) {
        Operation *definition = captures[i].getDefiningOp();
        if (!definition || !isa<wk::RootDomainOp, wk::DomainOp>(definition))
          return op.emitError("native helper captured an unresolved domain");
        if (auto domain = dyn_cast<wk::DomainOp>(definition)) {
          captures.insert(domain.getExtent());
          captures.insert(domain.getPartition());
        }
      }
    SmallVector<Value> inputs, sourceInputs;
    for (Value capture : captures) {
      if (isa<wk::DomainType>(capture.getType())) continue;
      if (carryTypes(capture.getType()).empty())
        return unsupported(op, "unsupported x86 helper capture: ") << capture.getType();
      sourceInputs.push_back(capture);
      llvm::append_range(inputs, carryValues(capture, op.getLoc()));
    }
    SmallVector<std::pair<std::string, Value>> shapeInputs;
    for (auto &entry : dimensions) shapeInputs.emplace_back(entry.getKey().str(), entry.second);
    llvm::sort(shapeInputs, [](auto &a, auto &c) { return a.first < c.first; });
    for (auto &entry : shapeInputs) inputs.push_back(entry.second);
    for (const Point &point : activePoints) {
      inputs.push_back(point.base); inputs.push_back(point.active); inputs.push_back(point.partition);
    }
    auto results = flattenTypes(op.getResultTypes(), op);
    if (failed(results)) return failure();
    SmallVector<Type> argumentTypes;
    for (Value value : inputs) argumentTypes.push_back(value.getType());
    func::FuncOp function;
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToEnd(module.getBody());
      unsigned sequence = 0;
      std::string name;
      do { name = "__weft_helper_" + std::to_string(sequence++); }
      while (SymbolTable::lookupSymbolIn(module, name));
      function = func::FuncOp::create(b, op.getLoc(), name, b.getFunctionType(argumentTypes, *results));
      function->setAttr("weft.x86.cpu", b.getStringAttr(target.cpu));
      function->setAttr("weft.x86.vector_bits", b.getI32IntegerAttr(target.vectorBits));
      function->setAttr("weft.x86.vector_registers", b.getI32IntegerAttr(target.vectorRegisters()));
      function.setPrivate();
      function->setAttr("weft.x86.helper", b.getDictionaryAttr({
          b.getNamedAttr("name", b.getStringAttr(op.getName())),
          b.getNamedAttr("noinline", b.getBoolAttr(op.getNoinline()))}));
      Lowering nested(module, target);
      nested.sourceModule = sourceModule;
      nested.shapeSymbols = shapeSymbols;
      nested.cfgFunction = function;
      nested.sourceParameters = sourceParameters;
      nested.shapeBindings = shapeBindings;
      Block *entry = function.addEntryBlock();
      nested.b.setInsertionPointToStart(entry);
      unsigned cursor = 0;
      for (Value source : sourceInputs) nested.bindCarry(source, entry->getArguments(), cursor);
      for (auto &shape : shapeInputs) nested.dimensions[shape.first] = entry->getArgument(cursor++);
      for (const Point &point : activePoints) {
        nested.activePoints.push_back({point.domain, entry->getArgument(cursor),
            entry->getArgument(cursor + 1), entry->getArgument(cursor + 2)});
        cursor += 3;
      }
      if (failed(nested.block(op.getBody().front()))) return failure();
      if (nested.b.getInsertionBlock()) return op.emitError("native helper has a path without a return");
    }
    auto call = func::CallOp::create(b, op.getLoc(), function, inputs);
    bindRange(op.getResults(), call.getResults());
    return success();
  }
  bool arrayValue(Type type) const {
    return isa<wk::ValueType>(type) && isa_and_nonnull<MemRefType>(physicalType(type));
  }
  Type arrayElement(Type type) const {
    return packedType(type) ? Type(IntegerType::get(type.getContext(), 8)) : scalar(type);
  }
  Value recordBytes(Location loc, Value elements, Type type) {
    auto layout = weft::analysis::encodingBytes(sourceModule, packedType(type));
    assert(succeeded(layout) && "verified encoded storage");
    return arith::MulIOp::create(b, loc,
        arith::DivUIOp::create(b, loc, elements, indexConstant(loc, layout->elements())),
        indexConstant(loc, layout->bytes()));
  }
  SmallVector<Value> storageExtents(Value value, Location loc) {
    auto extents = arrayExtents(value, loc);
    if (packedType(value.getType())) extents.back() = recordBytes(loc, extents.back(), value.getType());
    return extents;
  }
  SmallVector<Value> arrayExtents(Value value, Location loc) {
    if (auto found = arrayShapes.find(value); found != arrayShapes.end()) return found->second;
    SmallVector<Value> extents;
    for (int64_t extent : shape(value.getType())) {
      if (extent >= 0) extents.push_back(indexConstant(loc, extent));
      else if (auto local = localExtents.find(extent); local != localExtents.end()) extents.push_back(local->second);
      else if (extent >= -static_cast<int64_t>(shapeSymbols.size()))
        extents.push_back(dimensions.at(cast<StringAttr>(shapeSymbols[-extent - 1]).getValue()));
      else llvm_unreachable("verified local shape must be bound before its Value");
    }
    return extents;
  }
  Value arrayLane(Location loc, Value storage, Value index) {
    if (isa<MemRefType>(storage.getType())) return memref::LoadOp::create(b, loc, storage, index);
    if (isa<VectorType>(storage.getType())) {
      Value integer = arith::IndexCastOp::create(b, loc, b.getI64Type(), index);
      return operation(loc, "llvm.extractelement", scalar(storage.getType()), {storage, integer});
    }
    return storage;
  }
  Value arrayMask(Location loc, Value storage, Value index) {
    Value value = arrayLane(loc, storage, index);
    if (value.getType().isInteger(8)) value = arith::TruncIOp::create(b, loc, b.getI1Type(), value);
    return value;
  }
  void storeArrayMask(Location loc, Value value, Value storage, Value index) {
    Value byte = arith::ExtUIOp::create(b, loc, b.getI8Type(), value);
    memref::StoreOp::create(b, loc, byte, storage, index);
  }
  Value arrayOperandIndex(Value operand, ArrayRef<int64_t> outputAxes, ValueRange coordinates, Location loc,
                          ValueRange outputExtents = {}) {
    auto ids = axes(operand.getType());
    auto extents = arrayExtents(operand, loc);
    SmallVector<Value> selected;
    for (auto [axis, extent] : llvm::zip(ids, extents)) {
      auto position = llvm::find(outputAxes, axis);
      assert(position != outputAxes.end() && "canonical axis alignment preserves operand axes");
      if (!outputExtents.empty() && extent == outputExtents[position - outputAxes.begin()]) {
        selected.push_back(coordinates[position - outputAxes.begin()]);
        continue;
      }
      Value singleton = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, extent, indexConstant(loc, 1));
      selected.push_back(arith::SelectOp::create(b, loc, singleton, indexConstant(loc, 0), coordinates[position - outputAxes.begin()]));
    }
    return weft::x86::arrayLinear(b, loc, extents, selected);
  }
  Value packArray(Location loc, Value storage, Type resultType) {
    if (isa<MemRefType>(resultType)) return storage;
    if (!isa<VectorType>(resultType)) {
      Value value = memref::LoadOp::create(b, loc, storage, indexConstant(loc, 0));
      if (resultType.isInteger(1) && value.getType().isInteger(8))
        value = arith::TruncIOp::create(b, loc, resultType, value);
      return value;
    }
    // The fresh buffer has exactly the result's complete physical domain.
    // Preserve that contiguous transfer instead of rebuilding a loop-carried
    // vector with a dynamic insertion for every element. Inactive payloads and
    // their byte validity were already initialized by the producing traversal.
    auto storageType = cast<MemRefType>(storage.getType());
    auto loadedType = cast<VectorType>(resultType).clone(storageType.getElementType());
    Value loaded = vector::LoadOp::create(b, loc, loadedType, storage, indexConstant(loc, 0));
    if (loadedType != resultType)
      loaded = arith::TruncIOp::create(b, loc, resultType, loaded);
    return loaded;
  }
  void bindArray(Value result, const weft::x86::ArrayStorage &array, Location loc) {
    values.map(result, packArray(loc, array.data, physicalType(result.getType())));
    masks[result] = packArray(loc, array.valid, physicalMask(result.getType()));
    arrayShapes[result] = array.extents;
    if (auto encoding = packedType(result.getType())) {
      auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
      arrayShapes[result].back() = arith::MulIOp::create(b, loc,
          arith::DivUIOp::create(b, loc, array.extents.back(), indexConstant(loc, layout->bytes())),
          indexConstant(loc, layout->elements()));
    }
    if (isa<MemRefType>(mapped(result).getType()) && b.getInsertionPoint() != b.getInsertionBlock()->begin())
      arrayBirths[result] = &*std::prev(b.getInsertionPoint());
  }
  DenseMap<Value, Operation *> arrayBirths;
  DenseMap<Value, std::pair<int64_t, Value>> bfloatGuards;
  FailureOr<SmallVector<Value>> scalarOperation(Operation &op, ValueRange operands, ValueRange validity) {
    auto outerValues = values; auto outerMasks = masks;
    auto restore = llvm::scope_exit([&] { values = std::move(outerValues); masks = std::move(outerMasks); });
    Block scratch;
    OperationState state(op.getLoc(), op.getName()); state.addAttributes(op.getAttrs());
    for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
      auto argument = scratch.addArgument(element(operand.getType()), op.getLoc());
      values.map(argument, operands[i]); masks[argument] = validity[i]; state.addOperands(argument);
    }
    for (Type type : op.getResultTypes()) state.addTypes(element(type));
    auto clone = Operation::create(state); scratch.push_back(clone);
    if (failed(lower(*clone))) return failure();
    SmallVector<Value> result;
    for (Value value : clone->getResults()) { result.push_back(mapped(value)); result.push_back(valid(value, op.getLoc())); }
    return result;
  }
  LogicalResult arrayPointwise(Operation &op) {
    Location loc = op.getLoc();
    Value domain = op.getNumResults() ? op.getResult(0) : op.getOperand(3);
    auto extents = arrayExtents(domain, loc); auto ids = axes(domain.getType());
    SmallVector<weft::x86::ArrayStorage> outputs;
    for (Value result : op.getResults()) outputs.push_back(weft::x86::allocateArray(b, loc, scalar(result.getType()), extents));
    if (failed(weft::x86::forArray(b, loc, extents, [&](Value linear, ValueRange coordinates) {
      SmallVector<Value> operands, validity;
      for (Value operand : op.getOperands()) {
        Value index = arrayOperandIndex(operand, ids, coordinates, loc, extents);
        operands.push_back(arrayLane(loc, mapped(operand), index));
        validity.push_back(arrayMask(loc, valid(operand, loc), index));
      }
      auto result = scalarOperation(op, operands, validity);
      if (failed(result)) return failure();
      for (auto [i, output] : llvm::enumerate(outputs)) {
        memref::StoreOp::create(b, loc, (*result)[2 * i], output.data, linear);
        storeArrayMask(loc, (*result)[2 * i + 1], output.valid, linear);
      }
      return success();
    }, !isa<wk::PointerLoadOp, wk::PointerStoreOp, wk::PointerAtomicOp>(op), arrayScope(op)))) return failure();
    for (auto [result, output] : llvm::zip(op.getResults(), outputs)) bindArray(result, output, loc);
    return success();
  }
  LogicalResult arrayMemory(Operation &op, Value region, Value input = {}) {
    Location loc = op.getLoc(); Memory view = memories.at(region);
    Value domain = input ? input : op.getResult(0);
    auto extents = storageExtents(domain, loc); auto ids = axes(domain.getType()), memoryAxes = axes(region.getType());
    Type elementType = packedType(domain.getType()) ? Type(b.getIntegerType(8, IntegerType::Unsigned)) : element(domain.getType());
    if (auto encoding = packedType(domain.getType())) {
      auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
      view.offset = recordBytes(loc, view.offset, domain.getType());
      for (size_t axis = 0; axis + 1 < view.strides.size(); ++axis)
        view.strides[axis] = recordBytes(loc, view.strides[axis], domain.getType());
      view.extents.back() = recordBytes(loc, view.extents.back(), domain.getType());
      view.limits.back() = recordBytes(loc, view.limits.back(), domain.getType());
      view.storageMapped = constant(loc, b.getI1Type(), b.getBoolAttr(layout->rows != 1));
      view.storageMap = storageTable(loc, *layout);
      view.fieldElements = indexConstant(loc, layout->bytes());
    }
    weft::x86::ArrayStorage output;
    if (!input) output = weft::x86::allocateArray(b, loc, arrayElement(domain.getType()), extents);
    auto transfer = [&](Value linear, ValueRange coordinates, int64_t lanes) -> LogicalResult {
      Type valueType = arrayElement(domain.getType());
      Type addressType = b.getIndexType(), maskType = b.getI1Type(), accessType = elementType;
      if (lanes > 1) {
        valueType = VectorType::get({lanes}, valueType);
        addressType = VectorType::get({lanes}, b.getIndexType());
        maskType = VectorType::get({lanes}, b.getI1Type());
        accessType = wk::ValueType::get(b.getContext(), elementType,
            b.getDenseI64ArrayAttr({lanes}), b.getDenseI64ArrayAttr({ids.back()}));
      }
      auto splat = [&](Value value, Type type) -> Value {
        return lanes > 1 ? Value(b.createOrFold<vector::BroadcastOp>(loc, type, value)) : value;
      };
      Value offset = splat(view.offset, addressType), active = splat(view.guard, maskType);
      for (auto [axis, id] : llvm::enumerate(memoryAxes)) {
        auto position = llvm::find(ids, id);
        if (position == ids.end()) return op.emitError("x86 array memory lost a logical axis");
        Value coordinate = coordinates[position - ids.begin()];
        offset = arith::AddIOp::create(b, loc, offset, arith::MulIOp::create(b, loc, coordinate, splat(view.strides[axis], addressType)));
        // The owned array traversal already bounds every coordinate by its
        // logical extent, including packet tails and leading-axis gaps.
        Value extent = extents[position - ids.begin()];
        APInt extentConstant, limitConstant;
        bool full = !packedType(domain.getType()) && (extent == view.limits[axis] ||
            (matchPattern(extent, m_ConstantInt(&extentConstant)) &&
             matchPattern(view.limits[axis], m_ConstantInt(&limitConstant)) && extentConstant == limitConstant));
        if (!full) active = arith::AndIOp::create(b, loc, active,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, coordinate, splat(view.limits[axis], addressType)));
      }
      Value data = zero(loc, valueType);
      if (input) {
        Value validity;
        if (lanes > 1) {
          data = vector::LoadOp::create(b, loc, valueType, mapped(input), linear);
          auto bytesType = VectorType::get({lanes}, b.getI8Type());
          Value bytes = vector::LoadOp::create(b, loc, bytesType, valid(input, loc), linear);
          validity = arith::TruncIOp::create(b, loc, maskType, bytes);
        } else {
          data = arrayLane(loc, mapped(input), linear);
          validity = arrayMask(loc, valid(input, loc), linear);
        }
        active = arith::AndIOp::create(b, loc, active, validity);
      }
      Type indicesType = lanes > 1 ? Type(VectorType::get({lanes}, b.getI64Type())) : Type(b.getI64Type());
      offset = arith::IndexCastOp::create(b, loc, indicesType, offset);
      auto branch = scf::IfOp::create(b, loc, input ? TypeRange{} : TypeRange{data.getType()}, view.storageMapped, true);
      {
        OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&branch.getThenRegion().front());
        Value result = storageAccess(loc, view, offset, active, data, accessType, !input);
        if (!input) scf::YieldOp::create(b, loc, result);
        b.setInsertionPointToStart(&branch.getElseRegion().front());
        result = lanes > 1 ? accessRuntimeRuns(loc, view.base, offset, active, data, accessType, !input, lanes, view.strides.back()) :
                            access(loc, view.base, offset, active, data, accessType, !input, false);
        if (!input) scf::YieldOp::create(b, loc, result);
      }
      if (!input) {
        if (lanes > 1) {
          vector::StoreOp::create(b, loc, branch.getResult(0), output.data, linear);
          Value bytes = arith::ExtUIOp::create(b, loc, VectorType::get({lanes}, b.getI8Type()), active);
          vector::StoreOp::create(b, loc, bytes, output.valid, linear);
        } else {
          memref::StoreOp::create(b, loc, branch.getResult(0), output.data, linear);
          storeArrayMask(loc, active, output.valid, linear);
        }
      }
      return success();
    };
    Type elem = arrayElement(domain.getType());
    unsigned bits = elem.isIndex() ? 64 : elem.getIntOrFloatBitWidth();
    // A flat static traversal is suitable for LLVM only when its complete
    // storage span is contiguous. Row gaps or runtime leading strides otherwise
    // hide unit-stride inner rows behind division/remainder address expressions.
    bool dynamic = llvm::any_of(shape(domain.getType()), [](int64_t size) { return size < 0; });
    bool stridedRows = false;
    if (!dynamic && extents.size() > 1 && view.strides.size() == extents.size() &&
        matchPattern(view.strides.back(), m_One())) {
      auto dimensions = shape(domain.getType());
      int64_t stride = 1;
      for (int64_t axis = dimensions.size() - 1; axis >= 0; --axis) {
        APInt actual;
        if (!matchPattern(view.strides[axis], m_ConstantInt(&actual)) || actual.getSExtValue() != stride)
          stridedRows = true;
        stride *= dimensions[axis];
      }
    }
    bool packets = (dynamic || stridedRows) && !packedType(domain.getType()) && !extents.empty() && ids == memoryAxes &&
        bits >= 8 && bits <= 64 && llvm::isPowerOf2_32(bits) && matchPattern(view.storageMapped, m_Zero()) &&
        (!input || isa<MemRefType>(mapped(input).getType()));
    if (packets) {
      if (failed(weft::x86::forArrayPackets(b, loc, extents, target.vectorBits / bits, transfer))) return failure();
    } else if (failed(weft::x86::forArray(b, loc, extents,
        [&](Value linear, ValueRange coordinates) { return transfer(linear, coordinates, 1); }))) return failure();
    if (!input) { bindArray(op.getResult(0), output, loc); valueOrigins[op.getResult(0)] = view.origins; }
    return success();
  }
  LogicalResult arrayReduce(wk::ReduceOp op) {
    Location loc = op.getLoc(); auto inputShape = arrayExtents(op.getInput(), loc);
    SmallVector<Value> outputShape(inputShape); outputShape.erase(outputShape.begin() + op.getAxis());
    Type elem = scalar(op.getResult().getType()); bool fp = isa<FloatType>(elem), uns = unsignedType(op.getResult().getType());
    StringRef name = op.getKind() == "add" ? (fp ? "arith.addf" : "arith.addi") : op.getKind() == "max" ?
        (fp ? "arith.maxnumf" : uns ? "arith.maxui" : "arith.maxsi") : (fp ? "arith.minnumf" : uns ? "arith.minui" : "arith.minsi");
    auto neutral = weft::analysis::reductionIdentity(elem, op.getKind(), uns);
    if (!neutral) return unsupported(op, "unsupported x86 reduction identity: ") << op.getKind();
    Value identity = constant(loc, elem, neutral);
    bool packetReduction = (elem.isF32() || elem.isF64() || (isa<IntegerType>(elem) && elem.getIntOrFloatBitWidth() >= 8)) &&
                           isa<MemRefType>(mapped(op.getInput()).getType());
    auto reducePacket = [&](ValueRange coordinates) {
      SmallVector<Value> inputCoordinates(coordinates);
      inputCoordinates.insert(inputCoordinates.begin() + op.getAxis(), indexConstant(loc, 0));
      Value base = weft::x86::arrayLinear(b, loc, inputShape, inputCoordinates);
      Value stride = weft::x86::arraySize(b, loc, ValueRange(inputShape).drop_front(op.getAxis() + 1));
      auto kind = op.getKind() == "add" ? vector::CombiningKind::ADD : op.getKind() == "max" ?
          (fp ? vector::CombiningKind::MAXNUMF : uns ? vector::CombiningKind::MAXUI : vector::CombiningKind::MAXSI) :
          (fp ? vector::CombiningKind::MINNUMF : uns ? vector::CombiningKind::MINUI : vector::CombiningKind::MINSI);
      return weft::x86::reduceArrayPackets(b, loc, mapped(op.getInput()), valid(op.getInput(), loc),
          base, stride, inputShape[op.getAxis()], identity, kind, name, target.vectorBits, arrayScope(*op));
    };
    // A closed reduction with no surviving axis is already a scalar SSA pair.
    // Materializing a one-element payload/mask array adds no source lifetime.
    if (packetReduction && outputShape.empty()) {
      auto [value, active] = reducePacket({});
      values.map(op.getResult(), value); masks[op.getResult()] = active;
      return success();
    }
    auto output = weft::x86::allocateArray(b, loc, elem, outputShape);
    if (failed(weft::x86::forArray(b, loc, outputShape, [&](Value outputIndex, ValueRange coordinates) {
      if (packetReduction) {
        auto [value, active] = reducePacket(coordinates);
        memref::StoreOp::create(b, loc, value, output.data, outputIndex);
        storeArrayMask(loc, active, output.valid, outputIndex);
        return success();
      }
      auto reduction = scf::ForOp::create(b, loc, indexConstant(loc, 0), inputShape[op.getAxis()], indexConstant(loc, 1),
          ValueRange{identity, constant(loc, b.getI1Type(), b.getBoolAttr(false))});
      {
        OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(reduction.getBody());
        SmallVector<Value> inputCoordinates(coordinates);
        inputCoordinates.insert(inputCoordinates.begin() + op.getAxis(), reduction.getInductionVar());
        Value index = weft::x86::arrayLinear(b, loc, inputShape, inputCoordinates);
        Value active = arrayMask(loc, valid(op.getInput(), loc), index);
        Value input = arith::SelectOp::create(b, loc, active, arrayLane(loc, mapped(op.getInput()), index), identity);
        Value next = operation(loc, name, elem, {reduction.getRegionIterArgs()[0], input});
        Value any = arith::OrIOp::create(b, loc, reduction.getRegionIterArgs()[1], active);
        scf::YieldOp::create(b, loc, ValueRange{next, any});
      }
      memref::StoreOp::create(b, loc, reduction.getResult(0), output.data, outputIndex);
      storeArrayMask(loc, reduction.getResult(1), output.valid, outputIndex);
      return success();
    }))) return failure();
    bindArray(op.getResult(), output, loc); return success();
  }
  LogicalResult lowerArray(Operation &op) {
    if (isa<wk::ValueFactOp>(op)) return success();
    if (auto field = dyn_cast<wk::FieldOp>(op)) return arrayField(field);
    if (auto table = dyn_cast<wk::LookupOp>(op)) return arrayLookup(table);
    if (auto reshape = dyn_cast<wk::ReshapeOp>(op)) return arrayReshape(reshape);
    if (auto projection = dyn_cast<wk::ExtractOp>(op))
      return arrayProjection(op, projection.getInput(), projection.getIndices(), projection.getSelectors());
    if (auto projection = dyn_cast<wk::UpdateOp>(op))
      return arrayProjection(op, projection.getInput(), projection.getIndices(), projection.getSelectors(), projection.getValue());
    if (auto empty = dyn_cast<wk::EmptyOp>(op)) {
      SmallVector<Value> extents;
      for (Value extent : empty.getExtents()) extents.push_back(mapped(extent));
      auto output = weft::x86::allocateArray(b, op.getLoc(), scalar(empty.getResult().getType()), extents);
      if (failed(weft::x86::forArray(b, op.getLoc(), extents, [&](Value index, ValueRange) {
        storeArrayMask(op.getLoc(), constant(op.getLoc(), b.getI1Type(), b.getBoolAttr(true)), output.valid, index);
        return success();
      }))) return failure();
      bindArray(empty.getResult(), output, op.getLoc()); return success();
    }
    if (auto load = dyn_cast<wk::AdmitOp>(op)) return arrayMemory(op, load.getRegion());
    if (auto store = dyn_cast<wk::CommitOp>(op)) return arrayMemory(op, store.getRegion(), store.getValue());
    if (auto reduction = dyn_cast<wk::ReduceOp>(op)) return arrayReduce(reduction);
    if (isa<wk::DotOp, wk::ContractOp, wk::OuterContractOp>(op)) return arrayContraction(op);
    if (isa<wk::MaterializeOp>(op) || (isa<wk::NewOp>(op) && op.getNumOperands())) {
      Value input = op.getOperand(0), result = op.getResult(0);
      if (input.getType() == result.getType()) {
        values.map(result, mapped(input)); masks[result] = valid(input, op.getLoc());
        arrayShapes[result] = arrayExtents(input, op.getLoc()); return success();
      }
    }
    if (isa<wk::ConstantOp, arith::ConstantOp, wk::UnaryOp, wk::BinaryOp, wk::CompareOp, wk::SelectOp,
            wk::CastOp, wk::BitcastOp, wk::WidenOp, wk::NarrowOp, wk::NewOp,
            wk::PointerLoadOp, wk::PointerStoreOp, wk::PointerAtomicOp>(op)) return arrayPointwise(op);
    return unsupported(op, "x86 dynamic array lowering is not implemented for ") << op.getName();
  }
  LogicalResult arrayField(wk::FieldOp op) {
    Location loc = op.getLoc();
    auto layout = weft::analysis::encodingBytes(sourceModule, packedType(op.getOwner().getType()));
    if (failed(layout)) return unsupported(op, "x86 encoded array has no closed field layout");
    auto found = llvm::find(layout->base.getFieldNames(), b.getStringAttr(op.getName()));
    if (found == layout->base.getFieldNames().end()) return op.emitError("field is absent from its Encoding");
    unsigned fieldIndex = found - layout->base.getFieldNames().begin();
    auto fieldShape = cast<DenseI64ArrayAttr>(layout->base.getFieldShapes()[fieldIndex]).asArrayRef();
    int64_t fieldElements = volume(fieldShape);
    Type elementType = scalar(op.getResult().getType());
    unsigned width = elementType.getIntOrFloatBitWidth();
    int64_t slots = (width + 7) / 8 + 1;
    Value table = storageTable(loc, *layout, fieldIndex, false);
    auto extents = arrayExtents(op.getResult(), loc);
    auto output = weft::x86::allocateArray(b, loc, elementType, extents);
    if (failed(weft::x86::forArray(b, loc, extents, [&](Value linear, ValueRange) {
      Value record = arith::DivUIOp::create(b, loc, linear, indexConstant(loc, fieldElements));
      Value entry = arith::RemUIOp::create(b, loc, linear, indexConstant(loc, fieldElements));
      Value base = arith::MulIOp::create(b, loc, record, indexConstant(loc, layout->bytes()));
      Value bits = zero(loc, b.getI64Type()), active = constant(loc, b.getI1Type(), b.getBoolAttr(true));
      for (int64_t part = 0; part < slots; ++part) {
        SmallVector<Value> fragment;
        for (int64_t component = 0; component < 4; ++component) {
          Value address = arith::AddIOp::create(b, loc,
              arith::MulIOp::create(b, loc, entry, indexConstant(loc, slots * 5)),
              indexConstant(loc, part * 5 + component));
          address = arith::IndexCastOp::create(b, loc, b.getI64Type(), address);
          fragment.push_back(access(loc, table, address, constant(loc, b.getI1Type(), b.getBoolAttr(true)),
              zero(loc, b.getI64Type()), b.getI64Type(), true, false));
        }
        Value offset = arith::AddIOp::create(b, loc, base,
            arith::IndexCastOp::create(b, loc, b.getIndexType(), fragment[0]));
        Value byte = arrayLane(loc, mapped(op.getOwner()), offset);
        Value wide = operation(loc, "arith.extui", b.getI64Type(), byte);
        Value selected = arith::AndIOp::create(b, loc, arith::ShRUIOp::create(b, loc, wide, fragment[1]), fragment[2]);
        bits = arith::OrIOp::create(b, loc, bits, arith::ShLIOp::create(b, loc, selected, fragment[3]));
        Value required = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, fragment[2], zero(loc, b.getI64Type()));
        active = arith::AndIOp::create(b, loc, active, arith::SelectOp::create(b, loc, required,
            arrayMask(loc, valid(op.getOwner(), loc), offset), constant(loc, b.getI1Type(), b.getBoolAttr(true))));
      }
      if (width != 64) bits = operation(loc, "arith.trunci", b.getIntegerType(width), bits);
      if (isa<FloatType>(elementType)) bits = operation(loc, "arith.bitcast", elementType, bits);
      memref::StoreOp::create(b, loc, bits, output.data, linear);
      storeArrayMask(loc, active, output.valid, linear);
      return success();
    }))) return failure();
    bindArray(op.getResult(), output, loc);
    auto origin = origins(op.getOwner(), loc); Value recordOrigin = origin.back(); origin.pop_back();
    for (auto [i, extent] : llvm::enumerate(fieldShape))
      origin.push_back(i + 1 == fieldShape.size() && layout->elements() % extent == 0 ?
          Value(arith::DivUIOp::create(b, loc, recordOrigin, indexConstant(loc, layout->elements() / extent))) : indexConstant(loc, 0));
    valueOrigins[op.getResult()] = std::move(origin);
    return success();
  }
  LogicalResult arrayLookup(wk::LookupOp op) {
    Location loc = op.getLoc(); auto extents = arrayExtents(op.getIndices(), loc);
    auto output = weft::x86::allocateArray(b, loc, scalar(op.getResult().getType()), extents);
    if (failed(weft::x86::forArray(b, loc, extents, [&](Value linear, ValueRange) {
      Value index = arrayLane(loc, mapped(op.getIndices()), linear);
      if (!index.getType().isIndex()) index = arith::IndexCastUIOp::create(b, loc, b.getIndexType(), index);
      Value active = arrayMask(loc, valid(op.getIndices(), loc), linear);
      auto branch = scf::IfOp::create(b, loc, TypeRange{scalar(op.getResult().getType()), b.getI1Type()}, active, true);
      {
        OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&branch.getThenRegion().front());
        scf::YieldOp::create(b, loc, ValueRange{arrayLane(loc, mapped(op.getTable()), index), arrayMask(loc, valid(op.getTable(), loc), index)});
        b.setInsertionPointToStart(&branch.getElseRegion().front());
        scf::YieldOp::create(b, loc, ValueRange{zero(loc, scalar(op.getResult().getType())), constant(loc, b.getI1Type(), b.getBoolAttr(false))});
      }
      memref::StoreOp::create(b, loc, branch.getResult(0), output.data, linear);
      storeArrayMask(loc, branch.getResult(1), output.valid, linear);
      return success();
    }))) return failure();
    bindArray(op.getResult(), output, loc); return success();
  }
  LogicalResult arrayReshape(wk::ReshapeOp op) {
    Location loc = op.getLoc(); auto inputShape = arrayExtents(op.getInput(), loc), outputShape = arrayExtents(op.getResult(), loc);
    auto inputAxes = axes(op.getInput().getType());
    auto permutation = weft::analysis::axisPermutation(inputAxes, op.getOrder());
    if (!permutation) return op.emitError("x86 reshape requires a complete axis permutation");
    auto output = weft::x86::allocateArray(b, loc, scalar(op.getResult().getType()), outputShape);
    if (failed(weft::x86::forArray(b, loc, outputShape, [&](Value linear, ValueRange) {
      SmallVector<Value> coordinates(inputShape.size()); Value remaining = linear;
      for (unsigned position : llvm::reverse(*permutation)) {
        coordinates[position] = arith::RemUIOp::create(b, loc, remaining, inputShape[position]);
        remaining = arith::DivUIOp::create(b, loc, remaining, inputShape[position]);
      }
      Value index = weft::x86::arrayLinear(b, loc, inputShape, coordinates);
      memref::StoreOp::create(b, loc, arrayLane(loc, mapped(op.getInput()), index), output.data, linear);
      storeArrayMask(loc, arrayMask(loc, valid(op.getInput(), loc), index), output.valid, linear);
      return success();
    }))) return failure();
    bindArray(op.getResult(), output, loc); return success();
  }
  LogicalResult arrayProjection(Operation &op, Value source, ValueRange indices, ArrayAttr selectors, Value update = {}) {
    Location loc = op.getLoc(); Value result = op.getResult(0), projected = update ? update : result;
    auto sourceShape = storageExtents(source, loc), outputShape = storageExtents(projected, loc);
    auto sourceAxes = axes(source.getType()), outputAxes = axes(projected.getType());
    auto sourceOrigins = origins(source, loc);
    Type elementType = arrayElement(source.getType());
    SmallVector<Value> resultOrigins;
    for (int64_t id : outputAxes) {
      auto position = llvm::find(sourceAxes, id);
      resultOrigins.push_back(position == sourceAxes.end() ? indexConstant(loc, 0) : sourceOrigins[position - sourceAxes.begin()]);
    }
    SmallVector<Value> selectedOrigins(sourceAxes.size());
    unsigned indexCursor = 0;
    for (auto [axis, attribute] : llvm::enumerate(selectors)) {
      StringRef selector = cast<StringAttr>(attribute).getValue();
      if (selector == "all") continue;
      Value index = indices[indexCursor++];
      if (selector != "domain" && selector != "group_index") continue;
      auto found = points.find(index);
      if (found == points.end()) return op.emitError("x86 dynamic projection lost a Level point");
      const Point &point = found->second;
      selectedOrigins[axis] = selector == "domain" ? point.base :
          Value(arith::DivUIOp::create(b, loc, point.base, point.partition));
      if (selector == "domain")
        resultOrigins[llvm::find(outputAxes, sourceAxes[axis]) - outputAxes.begin()] = selectedOrigins[axis];
    }
    // A rectangular fixed tile of an addressable snapshot has contiguous rows.
    // Load those rows at the use, keeping the original snapshot storage and
    // per-axis tail predicates. Do not copy every byte into a fresh array merely
    // to read it back as a vector. Gather selectors and updates keep their
    // indexed traversal, since they do not establish this address relation.
    auto resultType = dyn_cast<VectorType>(physicalType(result.getType()));
    auto storageType = dyn_cast<MemRefType>(mapped(source).getType());
    auto lastSelector = selectors.size() == sourceAxes.size() ?
        cast<StringAttr>(selectors[selectors.size() - 1]).getValue() : StringRef("all");
    bool rectangular = !update && resultType && storageType && !outputAxes.empty() &&
        sourceAxes.back() == outputAxes.back() &&
        (lastSelector == "all" || lastSelector == "domain") &&
        llvm::none_of(selectors, [](Attribute attr) { return cast<StringAttr>(attr).getValue() == "gather"; });
    if (rectangular) {
      SmallVector<int64_t> tile = shape(projected.getType());
      if (packedType(source.getType())) {
        auto layout = weft::analysis::encodingBytes(sourceModule, packedType(source.getType()));
        tile.back() = tile.back() / layout->elements() * layout->bytes();
      }
      auto maskType = resultType.clone(b.getI1Type());
      Value payload = zero(loc, resultType), validity = zero(loc, maskType);
      int64_t rowWidth = tile.back();
      unsigned bits = storageType.getElementType().getIntOrFloatBitWidth();
      int64_t packetWidth = std::max<int64_t>(1, target.vectorBits / std::max<unsigned>(8, bits));
      for (int64_t row = 0; row < resultType.getNumElements() / rowWidth; ++row) {
        for (int64_t start = 0; start < rowWidth; start += packetWidth) {
          int64_t lanes = std::min(packetWidth, rowWidth - start);
          auto indicesType = VectorType::get({lanes}, b.getIndexType());
          auto predicateType = VectorType::get({lanes}, b.getI1Type());
          auto packetType = VectorType::get({lanes}, storageType.getElementType());
          auto byteType = VectorType::get({lanes}, b.getI8Type());
          SmallVector<int64_t> sequence;
          for (int64_t i = 0; i < lanes; ++i) sequence.push_back(i);
          Value offsets = arith::ConstantOp::create(b, loc, indicesType,
              DenseIntElementsAttr::get(indicesType, sequence));
          auto splatIndex = [&](Value value) -> Value {
            return vector::BroadcastOp::create(b, loc, indicesType, value);
          };
          Value active = constant(loc, predicateType, b.getBoolAttr(true));
          SmallVector<Value> selected;
          unsigned cursor = 0;
          for (auto [axis, id] : llvm::enumerate(sourceAxes)) {
            StringRef selector = axis < selectors.size() ? cast<StringAttr>(selectors[axis]).getValue() : "all";
            auto outputAxis = llvm::find(outputAxes, id);
            Value relative;
            if (selector == "all" || selector == "domain") {
              unsigned position = outputAxis - outputAxes.begin();
              int64_t coordinate = (row * rowWidth + start) /
                  volume(ArrayRef(tile).drop_front(position + 1)) % tile[position];
              relative = indexConstant(loc, coordinate);
            }
            Value coordinate;
            if (selector == "all") coordinate = relative;
            else {
              Value index = indices[cursor++];
              if (selector == "index") {
                coordinate = mapped(index);
                active = arith::AndIOp::create(b, loc, active,
                    vector::BroadcastOp::create(b, loc, predicateType, valid(index, loc)));
              } else {
                const Point &point = points.at(index);
                coordinate = arith::SubIOp::create(b, loc, selectedOrigins[axis], sourceOrigins[axis]);
                Value extent = point.active;
                if (packedType(source.getType()) && axis + 1 == sourceAxes.size()) {
                  coordinate = recordBytes(loc, coordinate, source.getType());
                  extent = recordBytes(loc, extent, source.getType());
                }
                if (selector == "domain") {
                  coordinate = arith::AddIOp::create(b, loc, coordinate, relative);
                  Value positions = splatIndex(relative);
                  if (axis + 1 == sourceAxes.size()) positions = arith::AddIOp::create(b, loc, positions, offsets);
                  active = arith::AndIOp::create(b, loc, active,
                      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, positions, splatIndex(extent)));
                }
              }
            }
            if (!coordinate.getType().isIndex())
              coordinate = arith::IndexCastUIOp::create(b, loc, b.getIndexType(), coordinate);
            selected.push_back(coordinate);
            Value positions = splatIndex(coordinate);
            if (axis + 1 == sourceAxes.size()) positions = arith::AddIOp::create(b, loc, positions, offsets);
            active = arith::AndIOp::create(b, loc, active,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, positions, splatIndex(sourceShape[axis])));
          }
          Value address = weft::x86::arrayLinear(b, loc, sourceShape, selected);
          Value any = vector::ReductionOp::create(b, loc, vector::CombiningKind::OR, active);
          address = arith::SelectOp::create(b, loc, any, address, indexConstant(loc, 0));
          Value data = vector::MaskedLoadOp::create(b, loc, packetType, mapped(source), ValueRange{address}, active, zero(loc, packetType));
          Value bytes = vector::MaskedLoadOp::create(b, loc, byteType, valid(source, loc), ValueRange{address}, active, zero(loc, byteType));
          Value mask = arith::TruncIOp::create(b, loc, predicateType, bytes);
          if (resultType.getElementType().isBF16()) {
            Value bits = arith::BitcastOp::create(b, loc, resultType.clone(b.getI16Type()), payload);
            Value chunk = arith::BitcastOp::create(b, loc, packetType.clone(b.getI16Type()), data);
            bits = vector::InsertStridedSliceOp::create(b, loc, chunk, bits,
                ArrayRef<int64_t>{row * rowWidth + start}, ArrayRef<int64_t>{1});
            payload = arith::BitcastOp::create(b, loc, resultType, bits);
          } else payload = vector::InsertStridedSliceOp::create(b, loc, data, payload,
              ArrayRef<int64_t>{row * rowWidth + start}, ArrayRef<int64_t>{1});
          validity = vector::InsertStridedSliceOp::create(b, loc, mask, validity,
              ArrayRef<int64_t>{row * rowWidth + start}, ArrayRef<int64_t>{1});
        }
      }
      values.map(result, payload); masks[result] = validity;
      valueOrigins[result] = resultOrigins;
      arrayShapes[result] = arrayExtents(result, loc);
      return success();
    }
    auto output = weft::x86::allocateArray(b, loc, elementType, update ? ValueRange(sourceShape) : ValueRange(outputShape));
    if (update && failed(weft::x86::forArray(b, loc, sourceShape, [&](Value index, ValueRange) {
      memref::StoreOp::create(b, loc, arrayLane(loc, mapped(source), index), output.data, index);
      storeArrayMask(loc, arrayMask(loc, valid(source, loc), index), output.valid, index);
      return success();
    }))) return failure();
    if (failed(weft::x86::forArray(b, loc, outputShape, [&](Value linear, ValueRange coordinates) -> LogicalResult {
      SmallVector<Value> selected;
      Value active = constant(loc, b.getI1Type(), b.getBoolAttr(true));
      unsigned cursor = 0;
      for (auto [axis, id] : llvm::enumerate(sourceAxes)) {
        StringRef selector = axis < selectors.size() ? cast<StringAttr>(selectors[axis]).getValue() : "all";
        auto outputAxis = llvm::find(outputAxes, id);
        Value coordinate;
        if (selector == "all") coordinate = coordinates[outputAxis - outputAxes.begin()];
        else {
          Value index = indices[cursor++];
          if (selector == "index") {
            coordinate = mapped(index);
            active = arith::AndIOp::create(b, loc, active, valid(index, loc));
          } else if (selector == "gather") {
            Value position = arrayOperandIndex(index, outputAxes, coordinates, loc);
            coordinate = arrayLane(loc, mapped(index), position);
            active = arith::AndIOp::create(b, loc, active, arrayMask(loc, valid(index, loc), position));
          } else if (selector == "domain" || selector == "group_index") {
            auto found = points.find(index);
            if (found == points.end()) return op.emitError("x86 dynamic projection lost a Level point");
            const Point &point = found->second;
            Value origin = selectedOrigins[axis];
            coordinate = arith::SubIOp::create(b, loc, origin, sourceOrigins[axis]);
            Value extent = point.active;
            if (packedType(source.getType()) && axis + 1 == sourceAxes.size()) {
              coordinate = recordBytes(loc, coordinate, source.getType());
              extent = recordBytes(loc, extent, source.getType());
            }
            if (selector == "domain") {
              Value offset = coordinates[outputAxis - outputAxes.begin()];
              coordinate = arith::AddIOp::create(b, loc, coordinate, offset);
              active = arith::AndIOp::create(b, loc, active,
                  arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, offset, extent));
            }
          } else return unsupported(op, "unsupported dynamic Value selector: ") << selector;
        }
        if (!coordinate.getType().isIndex()) coordinate = arith::IndexCastUIOp::create(b, loc, b.getIndexType(), coordinate);
        active = arith::AndIOp::create(b, loc, active,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, coordinate, sourceShape[axis]));
        selected.push_back(coordinate);
      }
      Value address = weft::x86::arrayLinear(b, loc, sourceShape, selected);
      if (update) {
        auto branch = scf::IfOp::create(b, loc, TypeRange{}, active, false);
        OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&branch.getThenRegion().front());
        memref::StoreOp::create(b, loc, arrayLane(loc, mapped(update), linear), output.data, address);
        storeArrayMask(loc, arrayMask(loc, valid(update, loc), linear), output.valid, address);
      } else {
        auto branch = scf::IfOp::create(b, loc, TypeRange{elementType, b.getI1Type()}, active, true);
        {
          OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&branch.getThenRegion().front());
          scf::YieldOp::create(b, loc, ValueRange{arrayLane(loc, mapped(source), address), arrayMask(loc, valid(source, loc), address)});
          b.setInsertionPointToStart(&branch.getElseRegion().front());
          scf::YieldOp::create(b, loc, ValueRange{zero(loc, elementType), constant(loc, b.getI1Type(), b.getBoolAttr(false))});
        }
        memref::StoreOp::create(b, loc, branch.getResult(0), output.data, linear);
        storeArrayMask(loc, branch.getResult(1), output.valid, linear);
      }
      return success();
    }))) return failure();
    bindArray(result, output, loc); valueOrigins[result] = update ? sourceOrigins : resultOrigins;
    return success();
  }
  LogicalResult arrayContraction(Operation &op) {
    auto selected = selectContractionEngine(op, true);
    if (failed(selected)) return failure();
    Location loc = op.getLoc(); Value result = op.getResult(0);
    auto outputShape = arrayExtents(result, loc);
    auto outputAxes = axes(result.getType());
    auto over = op.getAttrOfType<DenseI64ArrayAttr>("over").asArrayRef();
    auto leftAxes = axes(op.getOperand(0).getType());
    auto leftShape = arrayExtents(op.getOperand(0), loc);
    SmallVector<Value> reductionShape;
    for (int64_t axis : over) reductionShape.push_back(leftShape[llvm::find(leftAxes, axis) - leftAxes.begin()]);
    Type elementType = scalar(result.getType());
    auto output = weft::x86::allocateArray(b, loc, elementType, outputShape);
    auto generic = [&]() -> LogicalResult {
      return weft::x86::forArray(b, loc, outputShape, [&](Value outputIndex, ValueRange outputCoordinates) -> LogicalResult {
      auto reduction = scf::ForOp::create(b, loc, indexConstant(loc, 0), weft::x86::arraySize(b, loc, reductionShape), indexConstant(loc, 1),
          ValueRange{zero(loc, elementType), constant(loc, b.getI1Type(), b.getBoolAttr(false))});
      reduction->setAttr("weft.x86.array_contraction", b.getUnitAttr());
      reduction->setAttr("weft.x86.array_scope", arrayScope(op));
      {
        OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(reduction.getBody());
        Value linear = reduction.getInductionVar();
        SmallVector<Value> reducedCoordinates(reductionShape.size());
        for (int64_t axis = reductionShape.size() - 1; axis >= 0; --axis) {
          reducedCoordinates[axis] = axis == 0 ? linear :
              Value(arith::RemUIOp::create(b, loc, linear, reductionShape[axis]));
          if (axis) linear = arith::DivUIOp::create(b, loc, linear, reductionShape[axis]);
        }
        Value active = constant(loc, b.getI1Type(), b.getBoolAttr(true));
        SmallVector<Value> operands;
        for (Value operand : op.getOperands()) {
          auto ids = axes(operand.getType()); auto extents = arrayExtents(operand, loc);
          SmallVector<Value> coordinates;
          for (int64_t axis : ids) {
            auto position = llvm::find(outputAxes, axis);
            Value coordinate = position != outputAxes.end() ? outputCoordinates[position - outputAxes.begin()] :
                reducedCoordinates[llvm::find(over, axis) - over.begin()];
            // Shared contraction axes have equal extents by Canonical contract.
            // A free axis comes directly from this operand's output domain.
            // Missing axes broadcast by omission, not by an extent==1 select.
            coordinates.push_back(coordinate);
          }
          Value index = weft::x86::arrayLinear(b, loc, extents, coordinates);
          active = arith::AndIOp::create(b, loc, active, arrayMask(loc, valid(operand, loc), index));
          operands.push_back(castValue(loc, arrayLane(loc, mapped(operand), index), element(operand.getType()), element(result.getType())));
        }
        for (Value &operand : operands) operand = arith::SelectOp::create(b, loc, active, operand, zero(loc, elementType));
        Value sum = isa<FloatType>(elementType) ? operation(loc, "math.fma", elementType,
            {operands[0], operands[1], reduction.getRegionIterArgs()[0]}) :
            operation(loc, "arith.addi", elementType, {operation(loc, "arith.muli", elementType, operands), reduction.getRegionIterArgs()[0]});
        scf::YieldOp::create(b, loc, ValueRange{sum, arith::OrIOp::create(b, loc, reduction.getRegionIterArgs()[1], active)});
      }
      memref::StoreOp::create(b, loc, reduction.getResult(0), output.data, outputIndex);
      storeArrayMask(loc, reduction.getResult(1), output.valid, outputIndex);
      return success();
      });
    };
    auto rightAxes = axes(op.getOperand(1).getType());
    auto leftStatic = weft::x86::matrixOperandShape(op, 0, shape(op.getOperand(0).getType()));
    auto rightStatic = weft::x86::matrixOperandShape(op, 1, shape(op.getOperand(1).getType()));
    bool leftReductionFirst = weft::x86::matrixReductionFirst(op, 0);
    bool rightReductionFirst = weft::x86::matrixReductionFirst(op, 1);
    bool amx = *selected == weft::x86::ContractionEngine::MatrixBFloat;
    bool fixedRightMatrix = *selected == weft::x86::ContractionEngine::MatrixInteger &&
        scalar(op.getOperand(0).getType()).isInteger(8) &&
        isa<MemRefType>(mapped(op.getOperand(0)).getType()) &&
        isa<VectorType>(mapped(op.getOperand(1)).getType()) &&
        volume(shape(result.getType())) <= 4096;
    bool integerMatrix = *selected == weft::x86::ContractionEngine::MatrixInteger &&
        scalar(op.getOperand(0).getType()).isInteger(8) && scalar(op.getOperand(1).getType()).isInteger(8) &&
        isa<MemRefType>(mapped(op.getOperand(0)).getType()) && isa<MemRefType>(mapped(op.getOperand(1)).getType());
    if (integerMatrix || fixedRightMatrix) {
      Value complete = constant(loc, b.getI1Type(), b.getBoolAttr(true));
      for (Value source : op.getOperands()) {
        if (isa<VectorType>(mapped(source).getType())) {
          Value all = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, valid(source, loc));
          complete = arith::AndIOp::create(b, loc, complete, all);
          continue;
        }
        int64_t count = volume(shape(source.getType()));
        int64_t packet = std::gcd(int64_t(32), count);
        auto loop = scf::ForOp::create(b, loc, indexConstant(loc, 0), indexConstant(loc, count),
            indexConstant(loc, packet), ValueRange{complete});
        {
          OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(loop.getBody());
          Value bytes = vector::LoadOp::create(b, loc, VectorType::get({packet}, b.getI8Type()),
              valid(source, loc), loop.getInductionVar());
          Value predicates = arith::TruncIOp::create(b, loc, VectorType::get({packet}, b.getI1Type()), bytes);
          Value all = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, predicates);
          scf::YieldOp::create(b, loc, arith::AndIOp::create(b, loc, loop.getRegionIterArgs()[0], all).getResult());
        }
        complete = loop.getResult(0);
      }
      auto emitInteger = [&](bool masked) {
      weft::x86::ArrayStorage left{mapped(op.getOperand(0)), valid(op.getOperand(0), loc), leftShape};
      if (fixedRightMatrix) {
        Value right = mapped(op.getOperand(1));
        if (masked)
          right = arith::SelectOp::create(b, loc, valid(op.getOperand(1), loc), right, zero(loc, right.getType()));
        if (rightReductionFirst) {
          SmallVector<int64_t> order;
          for (int64_t column = 0; column < rightStatic[0]; ++column)
            for (int64_t term = 0; term < rightStatic[1]; ++term)
              order.push_back(term * rightStatic[0] + column);
          right = permute(loc, right, order, right.getType());
        }
        bool leftUnsigned = unsignedType(op.getOperand(0).getType());
        bool rightUnsigned = unsignedType(op.getOperand(1).getType());
        auto type = cast<VectorType>(right.getType());
        unsigned width = type.getElementTypeBitWidth();
        unsigned carrier = width > 8 ? 16 : 8;
        if (width != carrier) {
          right = operation(loc, rightUnsigned || width == 1 ? "arith.extui" : "arith.extsi",
                            type.clone(b.getIntegerType(carrier)), right);
          type = cast<VectorType>(right.getType());
        }
        auto emit = [&](Value rhs, const weft::x86::ArrayStorage &destination, bool isUnsigned) {
          weft::x86::materializeAMXIntegerFixed(module, b, loc, target, left, rhs, destination,
              leftStatic[0], rightStatic[0], leftStatic[1], leftUnsigned, isUnsigned, false,
              leftReductionFirst, masked);
        };
        if (carrier == 8) {
          emit(right, output, rightUnsigned);
        } else {
          auto bytes = type.clone(b.getI8Type());
          Value low = arith::TruncIOp::create(b, loc, bytes, right);
          Value high = operation(loc, rightUnsigned ? "arith.shrui" : "arith.shrsi", type,
                                 {right, constant(loc, type, b.getI16IntegerAttr(8))});
          high = arith::TruncIOp::create(b, loc, bytes, high);
          auto upper = weft::x86::allocateArray(b, loc, b.getI32Type(), arrayExtents(result, loc));
          emit(low, output, true);
          emit(high, upper, rightUnsigned);
          // Read each private result only after both tile computations finish.
          // Their packet-wise modular combination needs no wide live vector
          // across the second contraction's packing and tile transfers.
          int64_t count = leftStatic[0] * rightStatic[0];
          int64_t packet = std::gcd(int64_t(target.vectorBits / 32), count);
          auto payload = VectorType::get({packet}, b.getI32Type());
          auto combine = scf::ForOp::create(b, loc, indexConstant(loc, 0), indexConstant(loc, count),
                                           indexConstant(loc, packet));
          {
            OpBuilder::InsertionGuard combineGuard(b);
            b.setInsertionPointToStart(combine.getBody());
            Value offset = combine.getInductionVar();
            Value lo = vector::LoadOp::create(b, loc, payload, output.data, offset);
            Value hi = vector::LoadOp::create(b, loc, payload, upper.data, offset);
            hi = arith::ShLIOp::create(b, loc, hi, constant(loc, payload, b.getI32IntegerAttr(8)));
            Value sum = arith::AddIOp::create(b, loc, lo, hi);
            vector::StoreOp::create(b, loc, sum, output.data, offset);
          }
        }
      } else {
        weft::x86::ArrayStorage right{mapped(op.getOperand(1)), valid(op.getOperand(1), loc), arrayExtents(op.getOperand(1), loc)};
        weft::x86::materializeAMXInteger(module, b, loc, target, left, right, output,
            leftStatic[0], rightStatic[0], leftStatic[1], unsignedType(op.getOperand(0).getType()),
            unsignedType(op.getOperand(1).getType()), leftReductionFirst, rightReductionFirst, masked);
      }
      };
      auto choice = scf::IfOp::create(b, loc, TypeRange{}, complete, true);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&choice.getThenRegion().front());
      emitInteger(false);
      b.setInsertionPointToStart(&choice.getElseRegion().front());
      emitInteger(true);
      // Integer zero padding is exact, but does not make the output domain
      // complete. A result is active iff some original pair was active. This
      // computes only validity; partial inputs still execute the selected AMX
      // arithmetic and never introduce an alternate SIMD contraction.
      if (failed(weft::x86::forArray(b, loc, outputShape,
          [&](Value outputIndex, ValueRange coordinates) -> LogicalResult {
        auto loop = scf::ForOp::create(b, loc, indexConstant(loc, 0), indexConstant(loc, leftStatic[1]),
            indexConstant(loc, 1), ValueRange{constant(loc, b.getI1Type(), b.getBoolAttr(false))});
        loop->setAttr("weft.x86.matrix_validity", b.getUnitAttr());
        {
          OpBuilder::InsertionGuard loopGuard(b);
          b.setInsertionPointToStart(loop.getBody());
          SmallVector<Value> active;
          for (unsigned side = 0; side < 2; ++side) {
            Value source = op.getOperand(side);
            SmallVector<Value> point{coordinates[side], loop.getInductionVar()};
            if (side == 0 ? leftReductionFirst : rightReductionFirst) std::swap(point[0], point[1]);
            Value offset = weft::x86::arrayLinear(b, loc, arrayExtents(source, loc), point);
            active.push_back(arrayMask(loc, valid(source, loc), offset));
          }
          Value pair = arith::AndIOp::create(b, loc, active[0], active[1]);
          scf::YieldOp::create(b, loc, arith::OrIOp::create(b, loc, loop.getRegionIterArgs()[0], pair).getResult());
        }
        storeArrayMask(loc, loop.getResult(0), output.valid, outputIndex);
        return success();
      }))) return failure();
    } else if (amx) {
      weft::x86::ArrayStorage left{mapped(op.getOperand(0)), valid(op.getOperand(0), loc), leftShape};
      weft::x86::ArrayStorage right{mapped(op.getOperand(1)), valid(op.getOperand(1), loc), arrayExtents(op.getOperand(1), loc)};
      auto guardInput = [&](Value source, const weft::x86::ArrayStorage &input, int64_t count) {
        auto cached = bfloatGuards.find(source);
        if (cached != bfloatGuards.end() && cached->second.first == leftStatic[1]) return cached->second.second;
        OpBuilder::InsertionGuard guard(b);
        auto birth = arrayBirths.find(source);
        // Private SSA snapshots are immutable. Validate their initialized
        // payload once at birth, rather than again for every consumer tile.
        if (birth != arrayBirths.end()) b.setInsertionPointAfter(birth->second);
        Value safe = weft::x86::guardAMXBFloat(b, loc, input, count, leftStatic[1]);
        if (birth != arrayBirths.end()) bfloatGuards[source] = {leftStatic[1], safe};
        return safe;
      };
      Value leftSafe = guardInput(op.getOperand(0), left, leftStatic[0] * leftStatic[1]);
      Value rightSafe = guardInput(op.getOperand(1), right, rightStatic[0] * rightStatic[1]);
      Value safe = arith::AndIOp::create(b, loc, leftSafe, rightSafe);
      auto choice = scf::IfOp::create(b, loc, TypeRange{}, safe, true);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&choice.getThenRegion().front());
      weft::x86::materializeAMXBFloat(module, b, loc, target, left, right, output,
          leftStatic[0], rightStatic[0], leftStatic[1], leftReductionFirst, rightReductionFirst);
      b.setInsertionPointToStart(&choice.getElseRegion().front());
      if (failed(generic())) return failure();
    } else if (failed(generic())) return failure();
    bindArray(result, output, loc); return success();
  }
  LogicalResult lower(Operation &op) {
    if (auto admit = dyn_cast<wk::AdmitOp>(op); admit && selectIndexedSnapshot(admit)) return success();
    if (failed(lowerOperation(op))) return failure();
    if (!b.getInsertionBlock()) return success();
    for (Value result : op.getResults()) {
      if (!isa<wk::ValueType>(result.getType()) || valueOrigins.contains(result)) continue;
      SmallVector<Value> resultOrigins;
      for (auto [position, axis] : llvm::enumerate(axes(result.getType()))) {
        Value origin = indexConstant(op.getLoc(), 0);
        for (Value operand : op.getOperands()) {
          auto ids = axes(operand.getType());
          auto found = llvm::find(ids, axis);
          if (found == ids.end() || !isa<wk::ValueType>(operand.getType())) continue;
          origin = origins(operand, op.getLoc())[found - ids.begin()];
          if (isa<wk::MacGroupsOp, wk::Fold2Op>(op)) {
            int64_t group = isa<wk::Fold2Op>(op) ? 2 : cast<wk::MacGroupsOp>(op).getGroup();
            if (position + 1 == axes(result.getType()).size()) origin = arith::DivUIOp::create(b, op.getLoc(), origin, indexConstant(op.getLoc(), group));
          }
          break;
        }
        resultOrigins.push_back(origin);
      }
      valueOrigins[result] = std::move(resultOrigins);
    }
    for (Value result : op.getResults()) {
      if (!numeric(result.getType()) || masks.contains(result)) continue;
      Value mask = constant(op.getLoc(), physicalType(maskType(result.getType())), b.getBoolAttr(true));
      for (Value operand : op.getOperands()) {
        if (!numeric(operand.getType())) continue;
        auto inputAxes = axes(operand.getType()), outputAxes = axes(result.getType());
        if (!llvm::all_of(inputAxes, [&](int64_t axis) { return llvm::is_contained(outputAxes, axis); })) continue;
        mask = arith::AndIOp::create(b, op.getLoc(), mask, validAs(operand, result.getType(), op.getLoc()));
      }
      masks[result] = mask;
    }
    return success();
  }
  LogicalResult lowerOperation(Operation &op) {
    auto loc = op.getLoc();
    if (auto lookup = dyn_cast<wk::LookupOp>(op)) {
      auto found = indexedSnapshots.find(lookup.getTable());
      if (found != indexedSnapshots.end()) return indexedSnapshotLookup(lookup, found->second);
    }
    bool array = llvm::any_of(op.getOperandTypes(), [&](Type type) { return arrayValue(type); }) ||
                 llvm::any_of(op.getResultTypes(), [&](Type type) { return arrayValue(type); });
    if (array && op.getName().getDialectNamespace() == "weft_kernel" &&
        !isa<wk::HelperOp, wk::HelperReturnOp, wk::LevelOp, wk::HandoffOp, wk::BirthsYieldOp>(op)) return lowerArray(op);
    if (auto function = dyn_cast<wk::HelperOp>(op)) return helper(function);
    if (auto ret = dyn_cast<wk::HelperReturnOp>(op)) {
      if (!cfgFunction) return op.emitError("x86 helper return has no native function");
      func::ReturnOp::create(b, loc, flatten(ret.getValues(), loc));
      b.clearInsertionPoint();
      return success();
    }
    if (cfgFunction) {
      if (auto branch = dyn_cast<scf::IfOp>(op)) return cfgIf(branch);
      if (auto loop = dyn_cast<scf::ForOp>(op)) return cfgFor(loop);
      if (auto loop = dyn_cast<scf::WhileOp>(op)) return cfgWhile(loop);
    }
    if (isa<wk::RootDomainOp, wk::DomainOp, wk::EncodingDeclOp, wk::ValueFactOp>(op)) return success();
    if (auto symbol = dyn_cast<wk::SymbolOp>(op)) {
      if (symbol.getKind() == "source_auto" || (symbol.getKind() == "shape" && sourceParameters.contains(symbol.getName()))) {
        auto selected = sourceParameters.find(symbol.getName());
        if (selected == sourceParameters.end()) return op.emitError("unbound x86 source parameter");
        values.map(symbol.getResult(), indexConstant(loc, selected->second));
        return success();
      }
      Value value = dimensions.lookup(symbol.getName());
      if (!value || symbol.getKind() != "shape") return unsupported(op, "unbound x86 entry shape symbol");
      values.map(symbol.getResult(), value); return success();
    }
    if (auto region = dyn_cast<wk::SliceOp>(op)) return slice(region);
    if (auto projection = dyn_cast<wk::FieldOp>(op)) return field(projection);
    if (auto extent = dyn_cast<wk::LocalShapeOp>(op)) {
      values.map(extent.getResult(), mapped(extent.getExtent()));
      localExtents[-extent.getId()] = mapped(extent.getExtent()); return success();
    }
    if (auto empty = dyn_cast<wk::EmptyOp>(op)) {
      values.map(empty.getResult(), LLVM::UndefOp::create(b, loc, physicalType(empty.getResult().getType())));
      return success();
    }
    if (auto region = dyn_cast<wk::SubviewOp>(op)) return subview(region);
    if (auto size = dyn_cast<wk::ExtentOp>(op)) {
      values.map(size.getResult(), memories.at(size.getBase()).extents[size.getAxis()]); return success();
    }
    if (auto load = dyn_cast<wk::AdmitOp>(op)) return viewAccess(op, load.getRegion());
    if (auto store = dyn_cast<wk::CommitOp>(op)) return viewAccess(op, store.getRegion(), store.getValue());
    if (auto scope = dyn_cast<wk::LevelOp>(op)) return level(scope);
    if (auto ret = dyn_cast<wk::ReturnOp>(op)) {
      if (!ret.getValues().empty()) return ret.emitError("x86 entry cannot return values");
      func::ReturnOp::create(b, loc); return success();
    }
    if (isa<wk::ConstantOp, arith::ConstantOp>(op)) {
      Type type = physicalType(op.getResult(0).getType());
      if (!type) return unsupported(op, "unsupported x86 constant type");
      values.map(op.getResult(0), constant(loc, type, op.getAttr("value"))); return success();
    }
    if (auto iota = dyn_cast<wk::IotaOp>(op)) {
      auto type = cast<VectorType>(physicalType(iota.getResult().getType()));
      SmallVector<APInt> numbers;
      for (int64_t i = iota.getStart(); i < iota.getEnd(); ++i)
        numbers.emplace_back(type.getElementTypeBitWidth(), i);
      values.map(iota.getResult(), arith::ConstantOp::create(b, loc, type, DenseElementsAttr::get(type, numbers)));
      return success();
    }
    if (isa<wk::PointerCastOp>(op)) { values.map(op.getResult(0), mapped(op.getOperand(0))); return success(); }
    if (isa<wk::PointerLoadOp>(op)) return memory(op, true);
    if (isa<wk::PointerStoreOp>(op)) return memory(op, false);
    if (auto update = dyn_cast<wk::PointerAtomicOp>(op)) return atomic(update);
    if (auto bin = dyn_cast<wk::BinaryOp>(op)) return binary(bin);
    if (auto reduction = dyn_cast<wk::ReduceOp>(op)) return reduce(reduction);
    if (auto reduction = dyn_cast<wk::ReduceRegionOp>(op)) return regionReduction(reduction);
    if (auto prefix = dyn_cast<wk::ScanOp>(op)) return scan(prefix);
    if (auto mac = dyn_cast<wk::MacGroupsOp>(op)) return grouped(op, ValueRange{mac.getLhs(), mac.getRhs()}, mac.getGroup());
    if (auto fold = dyn_cast<wk::Fold2Op>(op)) return grouped(op, ValueRange{fold.getInput()}, 2);
    if (auto table = dyn_cast<wk::LookupOp>(op)) return lookup(table);
    if (isa<wk::DotOp, wk::ContractOp, wk::OuterContractOp>(op)) return contraction(op);
    if (isa<wk::CastOp, wk::WidenOp>(op)) {
      values.map(op.getResult(0), castValue(loc, mapped(op.getOperand(0)), op.getOperand(0).getType(), op.getResult(0).getType()));
      return success();
    }
    if (isa<wk::BitcastOp>(op)) {
      values.map(op.getResult(0), operation(loc, "arith.bitcast", physicalType(op.getResult(0).getType()), mapped(op.getOperand(0))));
      return success();
    }
    if (auto n = dyn_cast<wk::NarrowOp>(op)) {
      Type from = scalar(n.getInput().getType()), to = scalar(n.getResult().getType());
      Value input = mapped(n.getInput()), result;
      if (isa<IntegerType>(to)) {
        if (n.getSaturate()) {
          unsigned width = to.getIntOrFloatBitWidth();
          bool outputUnsigned = unsignedType(n.getResult().getType());
          APInt high = outputUnsigned ? APInt::getMaxValue(width) : APInt::getSignedMaxValue(width);
          APInt low = outputUnsigned ? APInt::getMinValue(width) : APInt::getSignedMinValue(width);
          if (auto fp = dyn_cast<FloatType>(from)) {
            Value upper = constant(loc, input.getType(), b.getFloatAttr(fp, high.roundToDouble(!outputUnsigned)));
            Value lower = constant(loc, input.getType(), b.getFloatAttr(fp, low.roundToDouble(!outputUnsigned)));
            input = operation(loc, "arith.minnumf", input.getType(), {input, upper});
            input = operation(loc, "arith.maxnumf", input.getType(), {input, lower});
          } else {
            unsigned bits = from.getIntOrFloatBitWidth();
            bool sourceUnsigned = unsignedType(n.getInput().getType());
            Value upper = constant(loc, input.getType(), b.getIntegerAttr(from, high.zext(bits)));
            input = operation(loc, sourceUnsigned ? "arith.minui" : "arith.minsi", input.getType(), {input, upper});
            if (!sourceUnsigned) {
              Value lower = constant(loc, input.getType(), b.getIntegerAttr(from, low.sext(bits)));
              input = operation(loc, "arith.maxsi", input.getType(), {input, lower});
            }
          }
        }
        if (isa<FloatType>(from)) {
          if (from.isF32() && to.getIntOrFloatBitWidth() >= 2 && to.getIntOrFloatBitWidth() <= 16 &&
              n.getRounding() == "dynamic") {
            // CVTPS2DQ performs the declared MXCSR rounding and conversion in
            // one instruction. Every defined narrow result fits signed i32;
            // the preceding saturation, when requested, stays in place.
            Value integer = dynamicRound(loc, input, true);
            values.map(n.getResult(), operation(loc, "arith.trunci", physicalType(n.getResult().getType()), integer));
            return success();
          }
          StringRef rounding = n.getRounding() == "rne" ? "math.roundeven" : n.getRounding() == "rdn" ? "math.floor" :
              n.getRounding() == "rup" ? "math.ceil" : "";
          if (!rounding.empty()) input = operation(loc, rounding, input.getType(), input);
          else if (n.getRounding() == "dynamic") input = dynamicRound(loc, input);
          else if (n.getRounding() != "rtz") return unsupported(n, "unsupported x86 float-to-integer rounding mode");
        }
        values.map(n.getResult(), castValue(loc, input, n.getInput().getType(), n.getResult().getType()));
        return success();
      }
      if (n.getRounding() != "rtz" && n.getRounding() != "rne")
        return unsupported(n, "unsupported x86 floating narrowing mode");
      if (from.isF32() && to.isF16()) {
        result = halfNarrow(loc, input, physicalType(n.getResult().getType()), n.getRounding() == "rtz");
      } else if (from.isF32() && to.isBF16()) {
        result = bfloatNarrow(loc, input, physicalType(n.getResult().getType()), n.getRounding() == "rtz");
      } else if (isa<FloatType>(from) && isa<FloatType>(to) && from.getIntOrFloatBitWidth() > to.getIntOrFloatBitWidth() &&
          n.getRounding() == "rtz") {
        Type narrowed = physicalType(n.getResult().getType());
        Value nearest = operation(loc, "arith.truncf", narrowed, input);
        Value restored = operation(loc, "arith.extf", input.getType(), nearest);
        Value absInput = operation(loc, "math.absf", input.getType(), input);
        Value absRestored = operation(loc, "math.absf", input.getType(), restored);
        Value overshot = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, absRestored, absInput);
        Type integer = b.getIntegerType(to.getIntOrFloatBitWidth());
        if (auto vector = dyn_cast<VectorType>(narrowed)) integer = VectorType::get(vector.getShape(), integer);
        Value bits = operation(loc, "arith.bitcast", integer, nearest);
        Value corrected = operation(loc, "arith.subi", integer, {bits, constant(loc, integer, b.getI64IntegerAttr(1))});
        corrected = operation(loc, "arith.bitcast", narrowed, corrected);
        result = arith::SelectOp::create(b, loc, overshot, corrected, nearest);
      } else {
        if (isa<FloatType>(from) && isa<IntegerType>(to) && n.getRounding() == "rne")
          input = operation(loc, "math.roundeven", input.getType(), input);
        result = castValue(loc, input, n.getInput().getType(), n.getResult().getType());
      }
      values.map(n.getResult(), result);
      return success();
    }
    if (auto select = dyn_cast<wk::SelectOp>(op)) {
      Type type = select.getResult().getType();
      values.map(select.getResult(), arith::SelectOp::create(b, loc,
          aligned(select.getCondition(), maskType(type), loc), aligned(select.getTrueValue(), type, loc), aligned(select.getFalseValue(), type, loc)));
      Value selected = arith::SelectOp::create(b, loc, aligned(select.getCondition(), maskType(type), loc),
          validAs(select.getTrueValue(), type, loc), validAs(select.getFalseValue(), type, loc));
      masks[select.getResult()] = arith::AndIOp::create(b, loc, selected, validAs(select.getCondition(), type, loc));
      return success();
    }
    if (auto cmp = dyn_cast<wk::CompareOp>(op)) {
      Type domain = cmp.getResult().getType();
      Type operand = cmp.getLhs().getType();
      Type alignedType = isa<wk::ValueType>(domain) ? Type(wk::ValueType::get(b.getContext(), element(operand),
          cast<wk::ValueType>(domain).getShape(), cast<wk::ValueType>(domain).getAxisIds())) : operand;
      Value a = aligned(cmp.getLhs(), alignedType, loc), c = aligned(cmp.getRhs(), alignedType, loc);
      StringRef p = cmp.getPredicate();
      if (p != "eq" && p != "ne" && p != "lt" && p != "le" && p != "gt" && p != "ge")
        return unsupported(cmp, "unsupported x86 comparison predicate: ") << p;
      Value result;
      if (isa<FloatType>(scalar(operand))) {
        auto predicate = p == "eq" ? arith::CmpFPredicate::OEQ : p == "ne" ? arith::CmpFPredicate::UNE :
            p == "lt" ? arith::CmpFPredicate::OLT : p == "le" ? arith::CmpFPredicate::OLE :
            p == "gt" ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OGE;
        result = arith::CmpFOp::create(b, loc, predicate, a, c);
      } else {
        bool u = unsignedType(operand);
        auto predicate = p == "eq" ? arith::CmpIPredicate::eq : p == "ne" ? arith::CmpIPredicate::ne :
            p == "lt" ? (u ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt) :
            p == "le" ? (u ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle) :
            p == "gt" ? (u ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt) :
                        (u ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge);
        result = arith::CmpIOp::create(b, loc, predicate, a, c);
      }
      values.map(cmp.getResult(), result); return success();
    }
    if (auto unary = dyn_cast<wk::UnaryOp>(op)) {
      StringRef kind = unary.getKind(), name;
      if (kind == "neg") name = isa<FloatType>(scalar(unary.getInput().getType())) ? "arith.negf" : "";
      else if (kind == "abs") name = isa<FloatType>(scalar(unary.getInput().getType())) ? "math.absf" : "math.absi";
      else if (kind == "exp") name = "math.exp";
      else if (kind == "exp2") name = "math.exp2";
      else if (kind == "log") name = "math.log";
      else if (kind == "log2") name = "math.log2";
      else if (kind == "sqrt") name = "math.sqrt";
      else if (kind == "rsqrt") name = "math.rsqrt";
      else if (kind == "sin") name = "math.sin";
      else if (kind == "cos") name = "math.cos";
      else if (kind == "floor") name = "math.floor";
      else if (kind == "ceil") name = "math.ceil";
      Type type = physicalType(unary.getResult().getType());
      Value input = mapped(unary.getInput()), result;
      if (kind == "neg" && name.empty()) result = operation(loc, "arith.subi", type, {zero(loc, type), input});
      else if (kind == "not") result = operation(loc, "arith.xori", type, {input, constant(loc, type, b.getI64IntegerAttr(-1))});
      else if (!name.empty()) result = operation(loc, name, type, input);
      else return unsupported(unary, "unsupported x86 unary operation: ") << kind;
      values.map(unary.getResult(), result); return success();
    }
    if (auto reshape = dyn_cast<wk::ReshapeOp>(op)) {
      auto dimensions = shape(reshape.getInput().getType()), inputAxes = axes(reshape.getInput().getType());
      auto permutation = weft::analysis::axisPermutation(inputAxes, reshape.getOrder());
      if (!permutation) return reshape.emitError("x86 reshape requires a complete axis permutation");
      const auto &order = *permutation;
      SmallVector<int64_t> reordered;
      for (auto d : order) reordered.push_back(dimensions[d]);
      SmallVector<int64_t> indices;
      for (int64_t lane = 0; lane < volume(dimensions); ++lane) {
        int64_t index = 0;
        for (auto [position, d] : llvm::enumerate(order))
          index += (lane / volume(ArrayRef(reordered).drop_front(position + 1)) % reordered[position]) *
                   volume(ArrayRef(dimensions).drop_front(d + 1));
        indices.push_back(index);
      }
      if (auto encoding = packedType(reshape.getInput().getType())) {
        auto layout = weft::analysis::encodingBytes(sourceModule, encoding);
        if (failed(layout)) return reshape.emitError("x86 encoded reshape has no storage mapping");
        int64_t elements = layout->elements(), bytes = layout->bytes();
        SmallVector<int64_t> byteIndices;
        for (int64_t first = 0; first < int64_t(indices.size()); first += elements) {
          if (first + elements > int64_t(indices.size()) || indices[first] % elements != 0)
            return reshape.emitError("x86 encoded reshape must preserve complete storage records");
          for (int64_t i = 0; i < elements; ++i)
            if (indices[first + i] != indices[first] + i)
              return reshape.emitError("x86 encoded reshape cannot permute elements inside a storage record");
          int64_t record = indices[first] / elements;
          for (int64_t byte = 0; byte < bytes; ++byte) byteIndices.push_back(record * bytes + byte);
        }
        // Encoded snapshots and their validity are byte-addressed. Logical
        // element indices would truncate or repeat bytes whenever the record's
        // storage span differs from its logical extent.
        values.map(reshape.getResult(), permute(loc, mapped(reshape.getInput()), byteIndices,
            physicalType(reshape.getResult().getType())));
        masks[reshape.getResult()] = permute(loc, valid(reshape.getInput(), loc), byteIndices,
            physicalMask(reshape.getResult().getType()));
        return success();
      }
      values.map(reshape.getResult(), permute(loc, mapped(reshape.getInput()), indices, physicalType(reshape.getResult().getType())));
      masks[reshape.getResult()] = permute(loc, valid(reshape.getInput(), loc), indices, physicalType(maskType(reshape.getResult().getType())));
      return success();
    }
    if (auto materialize = dyn_cast<wk::MaterializeOp>(op)) {
      if (packedType(materialize.getInput().getType())) {
        values.map(materialize.getResult(), mapped(materialize.getInput()));
        masks[materialize.getResult()] = valid(materialize.getInput(), loc);
        return success();
      }
      values.map(materialize.getResult(), aligned(materialize.getInput(), materialize.getResult().getType(), loc)); return success();
    }
    if (auto fresh = dyn_cast<wk::NewOp>(op)) {
      if (!fresh.getInitialized() || !fresh.getInitial())
        return unsupported(fresh, "unsupported x86 uninitialized local value");
      if (packedType(fresh.getResult().getType())) {
        values.map(fresh.getResult(), mapped(fresh.getInitial()));
        masks[fresh.getResult()] = valid(fresh.getInitial(), loc);
        return success();
      }
      values.map(fresh.getResult(), aligned(fresh.getInitial(), fresh.getResult().getType(), loc));
      return success();
    }
    if (auto extract = dyn_cast<wk::ExtractOp>(op))
      return projection(op, extract.getInput(), extract.getIndices(), extract.getSelectors());
    if (auto update = dyn_cast<wk::UpdateOp>(op))
      return projection(op, update.getInput(), update.getIndices(), update.getSelectors(), update.getValue());
    if (auto assume = dyn_cast<wk::AssumeOp>(op)) {
      LLVM::AssumeOp::create(b, loc, mapped(assume.getCondition())); return success();
    }
    // Ordinary SSA control stays in the common SCF dialect; only its carried
    // representation changes. No alternative Python/control-flow compiler.
    if (op.getName().getDialectNamespace() == "arith") {
      OperationState state(loc, op.getName());
      for (Value operand : op.getOperands()) state.addOperands(mapped(operand));
      for (Type type : op.getResultTypes()) state.addTypes(physicalType(type));
      state.addAttributes(op.getAttrs());
      values.map(op.getResults(), b.create(state)->getResults());
      return success();
    }
    if (op.getName().getDialectNamespace() == "scf") {
      OperationState state(loc, op.getName());
      // Conditions and induction bounds are structural operands, not carries.
      // Numeric carries pair their payload with validity even when scalar.
      unsigned structural = isa<scf::ForOp>(op) ? 3 : isa<scf::IfOp, scf::ConditionOp>(op) ? 1 : 0;
      for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
        if (index < structural) state.addOperands(mapped(operand));
        else state.addOperands(carryValues(operand, loc));
      }
      for (Type type : op.getResultTypes()) {
        auto converted = carryTypes(type);
        if (converted.empty()) return unsupported(op, "unsupported x86 control carry type: ") << type;
        state.addTypes(converted);
      }
      state.addAttributes(op.getAttrs());
      for (auto &region : op.getRegions()) state.addRegion();
      Operation *copy = b.create(state);
      unsigned cursor = 0;
      for (Value value : op.getResults()) bindCarry(value, copy->getResults(), cursor);
      OpBuilder::InsertionGuard guard(b);
      for (auto [source, dest] : llvm::zip(op.getRegions(), copy->getRegions())) {
        for (Block &old : source) {
          auto *next = new Block;
          dest.push_back(next);
          for (BlockArgument arg : old.getArguments()) {
            if (isa<scf::ForOp>(op) && arg.getArgNumber() == 0) {
              values.map(arg, next->addArgument(physicalType(arg.getType()), arg.getLoc()));
              continue;
            }
            auto types = carryTypes(arg.getType());
            if (types.empty()) return unsupported(op, "unsupported x86 region argument");
            SmallVector<Value> fields;
            for (Type type : types) fields.push_back(next->addArgument(type, arg.getLoc()));
            unsigned index = 0;
            bindCarry(arg, fields, index);
          }
          b.setInsertionPointToEnd(next);
          if (failed(block(old))) return failure();
        }
      }
      return success();
    }
    return unsupported(op, "unsupported x86 physical lowering for ") << op.getName();
  }
};

class ConvertToX86 : public PassWrapper<ConvertToX86, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToX86)
  explicit ConvertToX86(weft::X86Target target, std::string tuneCPU)
      : target(std::move(target)), tuneCPU(std::move(tuneCPU)) {}
  StringRef getArgument() const override { return "weft-convert-to-x86"; }
  void runOnOperation() override {
    ModuleOp source = getOperation();
    OwningOpRef<ModuleOp> physical = ModuleOp::create(source.getLoc());
    auto kernels = llvm::to_vector(source.getOps<wk::KernelOp>());
    if (kernels.size() != 1 || failed(Lowering(*physical, target).run(kernels.front()))) {
      signalPassFailure(); return;
    }
    if (!tuneCPU.empty())
      (*physical)->setAttr("weft.x86.tune_cpu", StringAttr::get(source.getContext(), tuneCPU));
    source->setAttrs((*physical)->getAttrs());
    source.getBodyRegion().takeBody(physical->getBodyRegion());
  }
private:
  weft::X86Target target;
  std::string tuneCPU;
};
} // namespace

void weft::registerX86Dialects(DialectRegistry &registry) {
  registry.insert<wk::WEFTKernelDialect, arith::ArithDialect, func::FuncDialect,
                  scf::SCFDialect, cf::ControlFlowDialect, vector::VectorDialect,
                  math::MathDialect, LLVM::LLVMDialect, memref::MemRefDialect,
                  bufferization::BufferizationDialect, ub::UBDialect>();
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    LLVM::MaskedLoadOp::attachInterface<TransferEffects<LLVM::MaskedLoadOp, true>>(*context);
    LLVM::masked_gather::attachInterface<TransferEffects<LLVM::masked_gather, true>>(*context);
    LLVM::MaskedStoreOp::attachInterface<TransferEffects<LLVM::MaskedStoreOp, false>>(*context);
    LLVM::masked_scatter::attachInterface<TransferEffects<LLVM::masked_scatter, false>>(*context);
  });
}

void weft::loadX86Dialects(MLIRContext &context) {
  DialectRegistry registry;
  registerX86Dialects(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<wk::WEFTKernelDialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect, cf::ControlFlowDialect, vector::VectorDialect,
                      math::MathDialect, LLVM::LLVMDialect, memref::MemRefDialect, bufferization::BufferizationDialect>();
  registerLLVMDialectTranslation(context);
}

void weft::registerX86Passes() {
  for (auto factory : {createX86MathPass, createX86HalfCarrierPass,
       createX86VectorProjectionPass, createX86ProjectionLoopPass,
       createX86IntegerReductionPass, createX86IntegerProductPass,
       createX86SnapshotLoadPass, createX86LookupProjectionPass,
       createX86IntegerPacketMergePass, createX86FullTilePass,
       createX86VectorCarryPass, createX86ArrayPacketPass,
       createX86ArrayForwardingPass, createX86ReductionSchedulePass,
       createX86UniformValidityPass, createX86IndexedValuePass,
       createX86IndexedPacketPass, createX86ContractionSchedulePass,
       createX86ReductionTreePass, createX86LocalStoragePass,
       createX86StreamingStorePass, createX86StreamPrefetchPass,
       x86::createAMXPackingReusePass, x86::createAMXSnapshotTransferPass,
       x86::createAMXConfigurationReusePass, x86::createAMXLoadSchedulePass})
    registerPass(factory);
  registerPass([] { return createX86SubbyteProjectionPass(); });
  registerPass([] { return createX86IntegerProjectionPass(); });
  registerPass([] { return createX86PipelineSchedulePass(); });
  registerPass([] { return createX86PipelinePass(); });
}

FailureOr<OwningOpRef<ModuleOp>> weft::lowerX86Module(ModuleOp source,
    const X86Target &target, const llvm::StringMap<int64_t> &bindings, llvm::StringRef tuneCPU,
    unsigned pipelineDepth) {
  if (pipelineDepth < 1 || pipelineDepth > 4) {
    source.emitError("pipeline_depth must be one of 1, 2, 3, 4"); return failure();
  }
  if (!tuneCPU.empty() && llvm::X86::parseTuneCPU(tuneCPU, true) == llvm::X86::CK_None) {
    source.emitError("unknown LLVM x86 tuning CPU: ") << tuneCPU;
    return failure();
  }
  loadX86Dialects(*source.getContext());
  OwningOpRef<ModuleOp> module = cast<ModuleOp>(source->clone());
  PassManager pm(source.getContext());
  SmallVector<Attribute> pipelines;
  auto recordPipeline = [&](PassManager &manager) {
    std::string pipeline;
    llvm::raw_string_ostream stream(pipeline);
    manager.printAsTextualPipeline(stream);
    pipelines.push_back(StringAttr::get(source.getContext(), pipeline));
  };
  pm.addPass(createBindSourceParametersPass(bindings));
  pm.addPass(std::make_unique<ConvertToX86>(target, tuneCPU.str()));
  pm.addPass(createX86MathPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(arith::createIntRangeOptimizationsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createX86UniformValidityPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86ArrayPacketPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86UniformValidityPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86ArrayForwardingPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86ContractionSchedulePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86ReductionSchedulePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86ProjectionLoopPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86VectorProjectionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86IndexedValuePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86SnapshotLoadPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86IntegerReductionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86ReductionTreePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86SubbyteProjectionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86IntegerProjectionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86FullTilePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86IndexedPacketPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(arith::createIntRangeOptimizationsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createX86UniformValidityPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Full-tile proofs remove guarded field/lookup regions. Only then can the
  // complete consumer cohort share one block with its integer decode chain.
  pm.addPass(createX86IntegerProjectionPass(true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86SubbyteProjectionPass(true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86LookupProjectionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86IntegerProjectionPass(true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86SubbyteProjectionPass(true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86IntegerPacketMergePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86HalfCarrierPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createX86IntegerProductPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(weft::x86::createAMXPackingReusePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(weft::x86::createAMXPackingReusePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(weft::x86::createAMXSnapshotTransferPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(weft::x86::createAMXConfigurationReusePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(weft::x86::createAMXLoadSchedulePass());
  pm.addPass(createX86StreamPrefetchPass());
  pm.addPass(createX86StreamingStorePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Folding input validity can make both numerical implementation branches
  // initialize uniform output validity. Close that proof before storage.
  pm.addPass(createX86UniformValidityPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (pipelineDepth > 1) {
    // Preserve iteration-private allocation identity until versioning is done;
    // stack slot reuse must analyze the expanded lifetimes, not the source loop.
    pm.addPass(createX86PipelineSchedulePass(pipelineDepth));
    pm.addPass(createX86PipelinePass());
    pm.addPass(createCanonicalizerPass());
  }
  pm.addPass(createX86LocalStoragePass());
  pm.addPass(createCanonicalizerPass());
  recordPipeline(pm);
  if (failed(pm.run(*module))) return failure();
  bool arrays = false;
  module->walk([&](memref::AllocOp) { arrays = true; });
  if (arrays) {
    PassManager ownership(source.getContext());
    bufferization::BufferDeallocationPipelineOptions options;
    options.privateFunctionDynamicOwnership = true;
    bufferization::buildBufferDeallocationPipeline(ownership, options);
    ownership.addPass(createCanonicalizerPass());
    recordPipeline(ownership);
    if (failed(ownership.run(*module))) return failure();
    (*module)->setAttr("weft.x86.dynamic_storage", UnitAttr::get(source.getContext()));
  }
  // Freeze control-flow representation in the physical module. In particular,
  // non-power-of-two vector carries must not become scalar register transfers
  // in SelectionDAG; the terminal LLVM translation only spells this decision.
  PassManager control(source.getContext());
  control.addPass(createSCFToControlFlowPass());
  control.addPass(createX86VectorCarryPass());
  recordPipeline(control);
  if (failed(control.run(*module))) return failure();
  (*module)->setAttr("weft.x86.pipelines", ArrayAttr::get(source.getContext(), pipelines));
  return module;
}

FailureOr<std::string> weft::translateX86Module(ModuleOp physical) {
  auto abi = physical->getAttrOfType<StringAttr>("weft.x86.abi");
  if (!abi) { physical.emitError("missing verified x86 physical ABI"); return failure(); }
  auto json = llvm::json::parse(abi.getValue());
  if (!json) { physical.emitError("invalid x86 physical ABI"); return failure(); }
  auto cpu = json->getAsObject()->getString("march");
  weft::X86Target selected;
  std::string error;
  if (!cpu || !weft::parseX86Target(cpu->str(), "sysv", 0, selected, error)) {
    physical.emitError("invalid x86 physical target: ") << error;
    return failure();
  }
  auto tuneCPU = physical->getAttrOfType<StringAttr>("weft.x86.tune_cpu");
  if (tuneCPU && llvm::X86::parseTuneCPU(tuneCPU.getValue(), true) == llvm::X86::CK_None) {
    physical.emitError("invalid x86 physical tuning CPU: ") << tuneCPU.getValue();
    return failure();
  }
  OwningOpRef<ModuleOp> module = cast<ModuleOp>(physical->clone());
  MLIRContext *context = physical.getContext();
  RewritePatternSet slices(context);
  vector::populateVectorInsertExtractStridedSliceTransforms(slices);
  if (failed(applyPatternsGreedily(*module, std::move(slices)))) return failure();
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  ub::populateUBToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  LLVMConversionTarget target(*context);
  target.addLegalOp<ModuleOp>();
  if (failed(applyFullConversion(*module, target, std::move(patterns)))) return failure();
  llvm::LLVMContext llvmContext;
  auto output = translateModuleToLLVMIR(*module, llvmContext);
  if (!output) return failure();
  output->setTargetTriple(llvm::Triple("x86_64-unknown-linux-gnu"));
  std::string features;
  for (const auto &feature : selected.features) {
    if (!features.empty()) features += ',';
    features += '+' + feature;
  }
  for (auto &function : *output)
    if (!function.isDeclaration()) {
      function.addFnAttr("target-cpu", selected.baseCPU());
      if (tuneCPU) function.addFnAttr("tune-cpu", tuneCPU.getValue());
      function.addFnAttr("target-features", features);
    }
  for (auto function : physical.getOps<func::FuncOp>())
    if (auto helper = function->getAttrOfType<DictionaryAttr>("weft.x86.helper")) {
      auto *native = output->getFunction(function.getSymName());
      if (!native) { function.emitError("native helper was lost during LLVM conversion"); return failure(); }
      native->setLinkage(llvm::GlobalValue::InternalLinkage);
      native->addFnAttr(cast<BoolAttr>(helper.get("noinline")).getValue() ?
          llvm::Attribute::NoInline : llvm::Attribute::AlwaysInline);
    }
  std::string result;
  llvm::raw_string_ostream stream(result);
  if (llvm::verifyModule(*output, &llvm::errs())) return failure();
  output->print(stream, nullptr);
  return result;
}
