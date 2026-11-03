#ifndef WEFT_COMPILER_H
#define WEFT_COMPILER_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define WEFT_COMPILER_API __declspec(dllexport)
#else
#define WEFT_COMPILER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* All handles and returned strings belong to the compiler library. No LLVM,
 * MLIR, Python or TVM object is part of this ABI. Input arrays are borrowed for
 * the duration of the call only. A module retains its compilation context. */
typedef struct weft_compiler_context weft_compiler_context;
typedef struct weft_compiler_module weft_compiler_module;

typedef struct weft_ir_value {
  int64_t id;
  const char *type;
} weft_ir_value;

typedef struct weft_ir_attribute {
  const char *name;
  const char *value;
} weft_ir_attribute;

typedef struct weft_ir_operation weft_ir_operation;
typedef struct weft_ir_region {
  const weft_ir_value *arguments;
  size_t argument_count;
  const weft_ir_operation *operations;
  size_t operation_count;
} weft_ir_region;

struct weft_ir_operation {
  const char *name;
  const char *filename;
  unsigned line;
  unsigned column;
  const int64_t *operands;
  size_t operand_count;
  const weft_ir_value *results;
  size_t result_count;
  const weft_ir_attribute *attributes;
  size_t attribute_count;
  const weft_ir_region *regions;
  size_t region_count;
};

/* Zero is success. On failure, error receives an owned diagnostic string.
 * Every non-null returned string is released with string_release. */
WEFT_COMPILER_API unsigned weft_compiler_abi(void);
WEFT_COMPILER_API void weft_compiler_string_release(char *text);
WEFT_COMPILER_API int weft_compiler_context_create(weft_compiler_context **result, char **error);
WEFT_COMPILER_API void weft_compiler_context_release(weft_compiler_context *context);
WEFT_COMPILER_API int weft_compiler_module_build(
    weft_compiler_context *context, const char *name,
    const weft_ir_operation *operations, size_t count,
    weft_compiler_module **result, char **error);
WEFT_COMPILER_API int weft_compiler_module_parse(
    weft_compiler_context *context, const char *text,
    weft_compiler_module **result, char **error);
WEFT_COMPILER_API void weft_compiler_module_release(weft_compiler_module *module);
WEFT_COMPILER_API int weft_compiler_module_verify(weft_compiler_module *module, char **error);
WEFT_COMPILER_API int weft_compiler_module_text(
    weft_compiler_module *module, char **result, char **error);
WEFT_COMPILER_API int weft_compiler_module_location_snapshot(
    weft_compiler_module *module, const char *filename, char **error);
WEFT_COMPILER_API int weft_compiler_module_interface(
    weft_compiler_module *module, char **result, char **error);
WEFT_COMPILER_API int weft_compiler_module_lower(
    weft_compiler_module *source, const char *options,
    weft_compiler_module **result, char **metadata, char **error);
WEFT_COMPILER_API int weft_compiler_module_emit(
    weft_compiler_module *module, char **result, char **error);
WEFT_COMPILER_API int weft_compiler_target_info(
    const char *query, const char *options, char **result, char **error);

#ifdef __cplusplus
}
#endif
#endif
