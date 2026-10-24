add_library(WeftAnalysis STATIC IntegerFacts.cpp)
target_link_libraries(WeftAnalysis PUBLIC WeftKernelDialect
  MLIRArithDialect MLIRVectorDialect MLIRIR)
