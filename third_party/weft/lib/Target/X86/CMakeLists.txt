add_library(WeftX86Target STATIC
  Backend/Compiler.cpp
  IR/Dialects.cpp
  Emission/LLVMIR.cpp
  Conversion/ConvertWeftToX86.cpp
  Transforms/Control/Pipeline.cpp
  Transforms/Contraction/ContractionEngines.cpp
  Transforms/Contraction/AMXContractions.cpp
  Transforms/Contraction/AMXTransfers.cpp
  Transforms/Memory/StreamPrefetch.cpp
  Transforms/Arithmetic/Math.cpp
  Transforms/Arithmetic/VectorProjections.cpp
  Transforms/Control/ProjectionLoops.cpp
  Transforms/Control/FullTiles.cpp
  Transforms/Memory/StreamingStores.cpp
  Transforms/Memory/ArrayPackets.cpp
  Transforms/Memory/ArrayForwarding.cpp
  Analysis/Resources.cpp
  Transforms/Control/ReductionSchedules.cpp
  Transforms/Arithmetic/UniformValidity.cpp
  Transforms/Memory/IndexedValues.cpp
  Transforms/Contraction/ContractionSchedules.cpp
  Transforms/Contraction/ContractionOutputs.cpp
  Transforms/Arithmetic/IntegerReductions.cpp
  Transforms/Arithmetic/Lookups.cpp
  Transforms/Resources/LocalStorage.cpp
  Analysis/StorageAnalysis.cpp
  Transforms/Memory/SnapshotLoads.cpp)
target_include_directories(WeftX86Target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(WeftX86Target PUBLIC WeftKernelDialect WeftX86Platform WeftSourceSpecialization WeftAnalysis WeftPhysicalTransforms
  LLVMTargetParser
  MLIRAnalysis MLIRArithDialect MLIRFuncDialect MLIRSCFDialect MLIRMathDialect MLIRVectorDialect
  MLIRLLVMDialect MLIRTransforms MLIRArithToLLVM MLIRMathToLLVM MLIRVectorToLLVM
  MLIRFuncToLLVM MLIRControlFlowToLLVM MLIRSCFToControlFlow MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation MLIRTargetLLVMIRExport MLIRUBToLLVM MLIRMathTransforms MLIRVectorTransforms
  MLIRMemRefDialect MLIRMemRefToLLVM MLIRBufferizationPipelines MLIRBufferizationTransforms
  MLIRSCFTransforms MLIRControlFlowTransforms MLIRArithTransforms MLIRViewLikeInterface)
