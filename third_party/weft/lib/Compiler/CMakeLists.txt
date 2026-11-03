add_library(WeftCompiler SHARED Compiler.cpp)
target_link_libraries(WeftCompiler PRIVATE WeftRISCVTarget WeftX86Target MLIRParser)
target_compile_options(WeftCompiler PRIVATE -fexceptions -fvisibility=hidden)
target_link_options(WeftCompiler PRIVATE "LINKER:--exclude-libs,ALL"
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_property(TARGET WeftCompiler APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exports.map)
set_target_properties(WeftCompiler PROPERTIES
  OUTPUT_NAME weft_compiler
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  INSTALL_RPATH "$ORIGIN")
install(TARGETS WeftCompiler LIBRARY DESTINATION ${WEFT_INSTALL_ROOT}/lib)
install(FILES ${WEFT_SOURCE_DIR}/runtime/include/weft/compiler.h
  DESTINATION ${WEFT_INSTALL_ROOT}/include/weft)
install(FILES ${WEFT_SOURCE_DIR}/cmake/WeftCompilerConfig.cmake
  DESTINATION ${WEFT_INSTALL_ROOT}/lib/cmake/WeftCompiler)
