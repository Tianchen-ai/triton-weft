add_library(WeftAnalysis STATIC IntegerFacts.cpp LogicalAxes.cpp)
target_link_libraries(WeftAnalysis PUBLIC WeftKernelDialect
  MLIRArithDialect MLIRVectorDialect MLIRIR)
