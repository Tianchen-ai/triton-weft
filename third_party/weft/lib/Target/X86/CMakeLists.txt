add_library(WeftX86Target STATIC
  Compiler.cpp
  ContractionEngines.cpp
  AMXContractions.cpp
  AMXTransfers.cpp
  StreamPrefetch.cpp
  Math.cpp
  VectorProjections.cpp
  ProjectionLoops.cpp
  FullTiles.cpp
  StreamingStores.cpp
  ArrayPackets.cpp
  ArrayForwarding.cpp
  Resources.cpp
  ReductionSchedules.cpp
  UniformValidity.cpp
  IndexedValues.cpp
  IndexRelations.cpp
  ContractionSchedules.cpp
  IntegerReductions.cpp
  Lookups.cpp
  LocalStorage.cpp
  SnapshotLoads.cpp)
target_link_libraries(WeftX86Target PUBLIC WeftKernelDialect WeftX86Platform WeftSourceSpecialization WeftAnalysis
  LLVMTargetParser
  MLIRAnalysis MLIRArithDialect MLIRFuncDialect MLIRSCFDialect MLIRMathDialect MLIRVectorDialect
  MLIRLLVMDialect MLIRTransforms MLIRArithToLLVM MLIRMathToLLVM MLIRVectorToLLVM
  MLIRFuncToLLVM MLIRControlFlowToLLVM MLIRSCFToControlFlow MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation MLIRTargetLLVMIRExport MLIRUBToLLVM MLIRMathTransforms MLIRVectorTransforms
  MLIRMemRefDialect MLIRMemRefToLLVM MLIRBufferizationPipelines MLIRBufferizationTransforms
  MLIRSCFTransforms MLIRControlFlowTransforms MLIRArithTransforms)
