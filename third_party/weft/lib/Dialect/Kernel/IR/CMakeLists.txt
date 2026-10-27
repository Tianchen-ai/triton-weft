add_mlir_dialect_library(WeftKernelDialect
  KernelDialect.cpp
  KernelTypes.cpp
  KernelInterface.cpp
  TypeQueries.cpp
  Verification.cpp
  EncodingOps.cpp
  ControlOps.cpp
  MemoryOps.cpp
  NumericOps.cpp

  DEPENDS
  WeftKernelIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces
)
