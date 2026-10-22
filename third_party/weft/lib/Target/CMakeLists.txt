add_library(WeftSourceSpecialization STATIC BindSourceParameters.cpp)
target_link_libraries(WeftSourceSpecialization PUBLIC WeftKernelDialect MLIRIR MLIRPass)

add_library(WeftRISCVTarget STATIC
  ${WEFT_RISCV_EXTENSION_EMISSION_SOURCES}
  RISCVPhysicalSupport.cpp
  RISCVCompilation.cpp
  RISCVCompilerOptions.cpp
  RISCVPipeline.cpp
  LocalMemory.cpp
  LocalStorageFootprint.cpp
  AlignRISCVLocalStorage.cpp
  ControlStorage.cpp
  PhysicalLoopControl.cpp
  DenseIssue.cpp
  DenseTransfer.cpp
  DenseProducerGraph.cpp
  ContractionSupply.cpp
  ConvertWeftToRISCV.cpp
  MaterializeRISCVReductions.cpp
  MaterializeRISCVAtomics.cpp
  FinalizeRISCVHelpers.cpp
  SelectRISCVOperations.cpp
  PropagateRISCVLayouts.cpp
  PlanRISCVResidency.cpp
  PlanRISCVMemory.cpp
  PointerMemory.cpp
  SpecializeRISCVMemory.cpp
  SelectRISCVDenseTransfers.cpp
  SelectRISCVFMAIssues.cpp
  ContractRISCVFloatingPoint.cpp
  SelectRISCVScalarPackets.cpp
  SimplifyRISCVControl.cpp
  PlanRISCVFieldStores.cpp
  CanonicalizeRISCVLayouts.cpp
  ConsumerLayouts.cpp
  FuseRISCVBitplanes.cpp
  LowerRISCVComposites.cpp
  MaterializeRISCVDenseFragmentLoads.cpp
  MaterializeRISCVLocalWindows.cpp
  MaterializeRISCVValuePrograms.cpp
  DenseValuePrograms.cpp
  DenseContractions.cpp
  DenseContractionEpilogue.cpp
  DensePanelTransfer.cpp
  SelectRISCVPanelTransfers.cpp
  PanelTransferSelection.cpp
  ValueLiveness.cpp
  LocalLifetime.cpp
  ForwardLocalCopies.cpp
  MaterializeRISCVPrograms.cpp
  VectorizeRISCVRecordLoops.cpp
  HoistRISCVLoopInvariants.cpp
  ScheduleRISCVLevels.cpp
  PipelineRISCVLevels.cpp
  PipelineShapeBindings.cpp
  PipelineSchedule.cpp
  PhysicalDiagnostics.cpp
  UnrollRISCVLevels.cpp
  ShareRISCVLayeredWindows.cpp
  Partial/Support.cpp
  Partial/AddTrees.cpp
  Partial/PlanTopologies.cpp
  Partial/Materialize.cpp
  FinalizeRISCVLeaves.cpp
  PredicateRegisters.cpp
  SelectRISCVScalarLoadPrimes.cpp
  ContractRISCVIntegerPointwise.cpp
  MaterializeRISCVReadSnapshots.cpp
  MaterializeRISCVResources.cpp
  FuseRISCVPhysicalIssueLoops.cpp
  EliminateDeadRISCVLayouts.cpp
  VerifyFinalRISCV.cpp
  RISCVIntrinsicC.cpp
  Emission/Control.cpp
  Emission/Functions.cpp
  Emission/ScalarControl.cpp
  Emission/Layout.cpp
  Emission/Memory.cpp
  Emission/RVVArithmetic.cpp
  Emission/Predicates.cpp
  Emission/VectorTiles.cpp
  Emission/FMAIssues.cpp
  Emission/Encoding.cpp
  Emission/FieldStorage.cpp
  Emission/RVVMemory.cpp
  Emission/IME.cpp
  IME/FragmentEmission.cpp
  IME/FragmentMaterialization.cpp
  IME/FragmentSupply.cpp
  Emission/RVVBitplanes.cpp
  Emission/RVVContraction.cpp
  Emission/RVVPartial.cpp
  Emission/Support.cpp
  RISCVCompiler.cpp
)

target_include_directories(WeftRISCVTarget PRIVATE ${WEFT_RISCV_EXTENSION_REGISTRY_DIR})
target_link_libraries(WeftRISCVTarget PUBLIC
  WeftSourceSpecialization
  WeftRISCVPlatform
  WeftRISCVContracts
  WeftKernelDialect
  WeftRISCVDialect
  MLIRArithDialect
  MLIRAsmParser
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
)

if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(WeftRISCVTarget USE_SHARED TargetParser)
else()
  llvm_config(WeftRISCVTarget TargetParser)
endif()

add_library(WeftX86Target STATIC
  X86/Compiler.cpp
  X86/AMXContractions.cpp
  X86/AMXTransfers.cpp
  X86/StreamPrefetch.cpp
  X86/Math.cpp
  X86/VectorProjections.cpp
  X86/ProjectionLoops.cpp
  X86/FullTiles.cpp
  X86/StreamingStores.cpp
  X86/ArrayPackets.cpp
  X86/ArrayForwarding.cpp
  X86/Resources.cpp
  X86/ReductionSchedules.cpp
  X86/UniformValidity.cpp
  X86/IndexedValues.cpp
  X86/IndexRelations.cpp
  X86/ContractionSchedules.cpp
  X86/IntegerReductions.cpp
  X86/Lookups.cpp
  X86/LocalStorage.cpp
  X86/SnapshotLoads.cpp)
target_link_libraries(WeftX86Target PUBLIC WeftKernelDialect WeftX86Platform WeftSourceSpecialization
  LLVMTargetParser
  MLIRAnalysis MLIRArithDialect MLIRFuncDialect MLIRSCFDialect MLIRMathDialect MLIRVectorDialect
  MLIRLLVMDialect MLIRTransforms MLIRArithToLLVM MLIRMathToLLVM MLIRVectorToLLVM
  MLIRFuncToLLVM MLIRControlFlowToLLVM MLIRSCFToControlFlow MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation MLIRTargetLLVMIRExport MLIRUBToLLVM MLIRMathTransforms MLIRVectorTransforms
  MLIRMemRefDialect MLIRMemRefToLLVM MLIRBufferizationPipelines MLIRBufferizationTransforms
  MLIRSCFTransforms MLIRControlFlowTransforms MLIRArithTransforms)
