#ifndef WEFT_GGML_H
#define WEFT_GGML_H

#include "weft/runtime.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_weft_counters {
    uint64_t nodes;
    uint64_t extra_buffer_nodes;
    uint64_t vec_dot_calls;
    uint64_t dot_products;
    uint64_t quantize_calls;
    uint64_t quantized_elements;
    uint64_t weft_quantize_calls;
} ggml_weft_counters;

/* GGML_WEFT is an explicit build choice. An enabled backend requires a native
   package in WEFT_GGML_PACKAGE; loading/ABI/execution failures are fatal GGML
   errors. A type absent from that package retains its upstream implementation. */
WEFT_API void ggml_weft_initialize(void);
WEFT_API uint64_t ggml_weft_ime1_calls(void);
WEFT_API uint64_t ggml_weft_ime2_calls(void);
WEFT_API uint64_t ggml_weft_amx_calls(void);
typedef struct ggml_weft_amx_binding {
    void (*function)(const uint8_t *, const uint8_t *, float *, size_t, size_t);
    void (*matrix_function)(const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t, size_t, size_t);
    void (*fixed_matrix)(const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t);
    void (*scratch_function)(const uint8_t *, const uint8_t *, float *, int16_t *, size_t, size_t);
    int16_t *scratch;
    uint64_t *calls;
    bool tile_state;
    void (*scratch_matrix)(const uint8_t *, const uint8_t *, float *, uint8_t *, size_t, size_t, size_t, size_t, size_t);
    uint8_t *matrix_scratch;
} ggml_weft_amx_binding;
WEFT_API ggml_weft_amx_binding ggml_weft_select_amx(enum ggml_type weight_type, bool prefill, size_t k);
static inline void ggml_weft_call_amx(ggml_weft_amx_binding binding, const uint8_t *w,
                                     const uint8_t *x, float *y, size_t n, size_t k) {
    if (binding.scratch_function) binding.scratch_function(w, x, y, binding.scratch, n, k);
    else binding.function(w, x, y, n, k);
    ++*binding.calls;
}
static inline void ggml_weft_call_amx_prefill(ggml_weft_amx_binding binding, const uint8_t *w,
                                             const uint8_t *x, float *y, size_t n, size_t k,
                                             size_t m, size_t stride) {
    if (binding.scratch_matrix) binding.scratch_matrix(w, x, y, binding.matrix_scratch, n, k, m, stride, 1);
    else if (binding.fixed_matrix) binding.fixed_matrix(w, x, y, k, stride, 1);
    else binding.matrix_function(w, x, y, n, k, m, stride, 1);
    ++*binding.calls;
}
static inline bool ggml_weft_amx_prefill_matches(ggml_weft_amx_binding binding, size_t m, size_t n) {
    return binding.matrix_function || (binding.scratch_matrix && m == 32) ||
           (binding.fixed_matrix && m == 32 && n == 32);
}
typedef size_t (*ggml_weft_ime1_t)(size_t, const uint8_t *, const uint8_t *, const uint8_t *,
                                 float *, size_t, size_t, size_t, size_t);
struct ggml_weft_ime1_binding;
/* The binding belongs to this worker and remains valid through its node. */
WEFT_API struct ggml_weft_ime1_binding *ggml_weft_select_ime1(ggml_weft_ime1_t original,
                                                          enum ggml_type weight_type);
WEFT_API size_t ggml_weft_call_ime1(struct ggml_weft_ime1_binding *, size_t,
                                  const uint8_t *, const uint8_t *, const uint8_t *,
                                  float *, size_t, size_t, size_t, size_t);
struct ggml_weft_ime2_binding;
WEFT_API struct ggml_weft_ime2_binding *ggml_weft_select_ime2(ggml_weft_ime1_t original,
                                                          enum ggml_type weight_type);
WEFT_API size_t ggml_weft_call_ime2(struct ggml_weft_ime2_binding *, size_t,
                                  const uint8_t *, const uint8_t *, const uint8_t *,
                                  float *, size_t, size_t, size_t, size_t);
WEFT_API bool ggml_weft_registered(enum ggml_type type);
WEFT_API bool ggml_weft_counters_get(enum ggml_type type, ggml_weft_counters *result);
WEFT_API const char *ggml_weft_kernel_name(enum ggml_type type);

/* CPU backend call-site hooks; callers preserve the original graph, worker
   assignment, quantization workspace, weight layout and synchronization. */
WEFT_API void ggml_weft_begin_node(const struct ggml_tensor *tensor, int ith, bool use_ref);
WEFT_API void ggml_weft_begin_graph(int node_count, struct ggml_tensor *const *nodes,
                                     bool external_callback);
WEFT_API void ggml_weft_end_node(void);
WEFT_API void ggml_weft_extra_node(const struct ggml_tensor *tensor, int ith);
typedef struct ggml_weft_dot_binding {
    ggml_vec_dot_t original;
    void (*function)(const uint8_t *, const uint8_t *, float *, size_t);
    uint64_t *calls;
} ggml_weft_dot_binding;
WEFT_API ggml_weft_dot_binding ggml_weft_select_vec_dot(enum ggml_type type, ggml_vec_dot_t original);

static inline void ggml_weft_call_dot(ggml_weft_dot_binding binding, int n, float *s,
                                      size_t bs, const void *x, size_t bx,
                                      const void *y, size_t by, int nrc) {
    if (binding.function) {
        binding.function((const uint8_t *)x, (const uint8_t *)y, s, (size_t)n);
        ++*binding.calls;
    } else {
        binding.original(n, s, bs, x, bx, y, by, nrc);
    }
}
WEFT_API void ggml_weft_from_float(ggml_from_float_t original, const float *x, void *y, int64_t n);
WEFT_API void ggml_weft_from_float_packed(enum ggml_type activation, ggml_from_float_t original,
                                        const float *x, void *y, int64_t n);

#ifdef __cplusplus
}
#endif
#endif
