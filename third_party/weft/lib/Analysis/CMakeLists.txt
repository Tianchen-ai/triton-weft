add_library(WeftAnalysis STATIC
  IntegerFacts.cpp
  IndexRelations.cpp
  LogicalAxes.cpp
  PhysicalDependence.cpp
  StorageBytes.cpp
  StorageLifetime.cpp)
target_link_libraries(WeftAnalysis PUBLIC WeftKernelDialect
  MLIRArithDialect MLIRSCFDialect MLIRVectorDialect MLIRIR MLIRSideEffectInterfaces
  MLIRInferIntRangeInterface)
