#include "Weft/Target/X86/X86Compiler.h"
#include "Weft/Dialect/Kernel/Transforms/SourceSpecialization.h"
#include "Conversion/Passes.h"
#include "Transforms/Contraction/AMXContractions.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/TargetParser/X86TargetParser.h"

using namespace mlir;

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
  pm.addPass(x86::createCanonicalConversionPass(target, tuneCPU));
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
