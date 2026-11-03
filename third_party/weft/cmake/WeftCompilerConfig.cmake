get_filename_component(WeftCompiler_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT TARGET Weft::Compiler)
  add_library(Weft::Compiler SHARED IMPORTED)
  set_target_properties(Weft::Compiler PROPERTIES
    IMPORTED_LOCATION "${WeftCompiler_ROOT}/lib/libweft_compiler.so"
    INTERFACE_INCLUDE_DIRECTORIES "${WeftCompiler_ROOT}/include")
endif()
set(WeftCompiler_FOUND TRUE)
