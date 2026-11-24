add_subdirectory(Target/RISCV/Extensions)
add_library(WeftRISCVContracts STATIC
  ${WEFT_RISCV_EXTENSION_CONTRACT_SOURCES}
  Target/RISCV/Platform/RISCVFragment.cpp
  Target/RISCV/Platform/RISCVExtensions.cpp
  Target/RISCV/Platform/RISCVDataCache.cpp
)
set_target_properties(WeftRISCVContracts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(WeftRISCVContracts PRIVATE ${WEFT_RISCV_EXTENSION_REGISTRY_DIR})

# Compiler and deployment share one ISA parser and native capability probe.
add_library(WeftRISCVPlatform STATIC Target/RISCV/Platform/RISCVTargetProfile.cpp Target/RISCV/Platform/RISCVISA.cpp
  Target/RISCV/Platform/RISCVTargetCatalog.cpp Target/RISCV/Platform/RISCVNativeTarget.cpp Target/RISCV/Platform/RISCVNativeCache.cpp)
set_target_properties(WeftRISCVPlatform PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(WeftRISCVPlatform PUBLIC WeftRISCVContracts)
if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(WeftRISCVPlatform USE_SHARED TargetParser Support)
else()
  llvm_config(WeftRISCVPlatform TargetParser Support)
endif()

if(NOT TRITON_WEFT_RUNTIME_ONLY)
  add_subdirectory(Dialect)
  add_subdirectory(Analysis)
  add_subdirectory(Transforms)
  add_subdirectory(Target)
endif()
add_library(WeftX86Platform STATIC Target/X86/Platform/Target.cpp Target/X86/Platform/OneDNN.cpp)
set_target_properties(WeftX86Platform PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(WeftX86Platform PUBLIC WEFT_RELEASE_VERSION="${WEFT_RELEASE_VERSION}")
target_compile_definitions(WeftX86Platform PRIVATE
  WEFT_ONEDNN_HEADER_AVAILABLE=${WEFT_ONEDNN_HEADER_AVAILABLE}
  WEFT_ONEDNN_ROOT_DIR="${WEFT_ONEDNN_ROOT}"
  WEFT_ONEDNN_LIBRARY_PATH="${WEFT_ONEDNN_LIBRARY_PATH}")
if(WEFT_ONEDNN_HEADER_AVAILABLE)
  target_include_directories(WeftX86Platform SYSTEM PRIVATE "${WEFT_ONEDNN_ROOT}/include")
endif()
target_link_libraries(WeftX86Platform PUBLIC ${CMAKE_DL_LIBS})
if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(WeftX86Platform USE_SHARED TargetParser Support)
else()
  llvm_config(WeftX86Platform TargetParser Support)
endif()
add_subdirectory(Runtime)
