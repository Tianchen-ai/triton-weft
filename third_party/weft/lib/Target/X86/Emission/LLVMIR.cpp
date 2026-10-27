#include "Weft/Target/X86/X86Compiler.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/X86TargetParser.h"

using namespace mlir;

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
