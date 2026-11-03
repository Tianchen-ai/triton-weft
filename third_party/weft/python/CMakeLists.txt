find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE WEFT_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 REQUIRED CONFIG PATHS ${WEFT_PYBIND11_DIR})
pybind11_add_module(_runtime MODULE runtime.cc)
target_compile_options(_runtime PRIVATE -fexceptions -frtti)
target_include_directories(_runtime PRIVATE ${WEFT_SOURCE_DIR}/runtime/include)
install(TARGETS _runtime LIBRARY DESTINATION weft_core)
install(DIRECTORY weft_core/ DESTINATION weft_core FILES_MATCHING PATTERN "*.py")
