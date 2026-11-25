"""Typed CPU launch code; grid coordinates stay outside canonical semantics."""
import copy


GRID_DECLARATION = """typedef int (*triton_check_thread)(const void *, char *, size_t);
int weft_kernel_take_status(char *error, size_t error_size);
void weft_kernel_release_hw_context(void);
int triton_weft_grid(void (*entry)(void), void *const *arguments,
                    size_t grid0, size_t grid1, size_t grid2, int threads,
                    const void *scope, triton_check_thread check, char *error, size_t error_size);"""


def single_instance_abi(abi):
    result = copy.deepcopy(abi)
    result["symbol"] += "_instance"
    result["arguments"] = [arg for arg in result["arguments"] if not arg["name"].startswith("__triton_")]
    parameters = [f"{arg['c_type']} {arg['name']}" for arg in result["arguments"]]
    extras = len(abi["shape_parameters"]) + sum(len(arg["shape"]) for arg in abi["arguments"] if arg["strided"])
    names = {arg["name"] for arg in result["arguments"]}
    for i in range(extras):
        name = f"extra_{i}"
        while name in names:
            name += "_"
        names.add(name)
        parameters.append(f"size_t {name}")
    result["declaration"] = f"void {result['symbol']}({', '.join(parameters) or 'void'});"
    return result


_WEFT_STATUS_SHIM_NOOP = r'''
int weft_kernel_take_status(char *error, size_t error_size) {
  if (error && error_size) error[0] = 0;
  return 0;
}
void weft_kernel_release_hw_context(void) {}
'''


_ONEDNN_BRGEMM_SHIM = r'''
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#if defined(__linux__) && defined(__x86_64__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <oneapi/dnnl/dnnl_ukernel.h>

static _Thread_local int __weft_tls_status = 0;
static _Thread_local char __weft_tls_error[512];
static _Thread_local const void *__weft_active_hw_brg = NULL;
static _Thread_local int __weft_amx_hw_active = 0;

void weft_kernel_release_hw_context(void) {
  if (__weft_amx_hw_active) {
    (void)dnnl_brgemm_release_hw_context();
    __weft_amx_hw_active = 0;
  }
  __weft_active_hw_brg = NULL;
}

static inline void __weft_reset_tls_status(void) {
  __weft_tls_status = 0;
  __weft_tls_error[0] = 0;
}

static void __weft_record_error(int status, const char *fmt, ...) {
  if (__weft_tls_status != 0) return;
  __weft_tls_status = status ? status : 6;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(__weft_tls_error, sizeof(__weft_tls_error), fmt, ap);
  va_end(ap);
}

int weft_kernel_take_status(char *error, size_t error_size) {
  int st = __weft_tls_status;
  if (error && error_size) {
    if (st != 0 && __weft_tls_error[0]) {
      strncpy(error, __weft_tls_error, error_size - 1);
      error[error_size - 1] = 0;
    } else {
      error[0] = 0;
    }
  }
  __weft_tls_status = 0;
  __weft_tls_error[0] = 0;
  return st;
}

static inline int __weft_status_from_dnnl(dnnl_status_t st, int fallback) {
  if (st == dnnl_invalid_arguments) return 1; /* WEFT_INVALID_ARGUMENT */
  if (st == dnnl_out_of_memory) return 6;     /* WEFT_RESOURCE_ERROR */
  if (st == dnnl_unimplemented) return 5;     /* WEFT_UNSUPPORTED */
  return fallback;
}

struct __weft_brgemm_entry {
  int64_t M, N, K, lda, ldb, ldc;
  int32_t dt_a, dt_b, dt_c, add_c;
  dnnl_brgemm_t brg;
  size_t scratch_bytes;
  uint64_t stamp;
  int valid;
};

static inline int __weft_map_dnnl_dt(int32_t code, dnnl_data_type_t *out) {
  switch (code) {
    case 0: *out = dnnl_f32; return 0;
    case 1: *out = dnnl_f16; return 0;
    case 2: *out = dnnl_bf16; return 0;
    case 3: *out = dnnl_s8; return 0;
    case 4: *out = dnnl_u8; return 0;
    case 5: *out = dnnl_s32; return 0;
    default: *out = dnnl_data_type_undef; return 1;
  }
}

void __weft_onednn_brgemm_execute(
    const void *A, const void *B, void *C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc,
    int32_t dt_a, int32_t dt_b, int32_t dt_c, int32_t add_c) {
  if (__weft_tls_status != 0) return;
  if (M < 0 || N < 0 || K < 0) {
    __weft_record_error(1, "Weft oneDNN BRGeMM received negative extent (M=%lld, N=%lld, K=%lld)",
                        (long long)M, (long long)N, (long long)K);
    return;
  }
  if (M == 0 || N == 0 || K == 0) return;
  if (!A || !B || !C) {
    __weft_record_error(1, "Weft oneDNN BRGeMM received null operand pointer");
    return;
  }
  /* L1/L2 K-blocking for large f32 BRGeMM contractions keeps C hot in L1 and
   * the active A/B slabs resident in L1/L2 across all K blocks. */
  if (dt_a == 0 && dt_b == 0 && dt_c == 0 && K > 512) {
    const int64_t k_blk = 512;
    const float *af = (const float *)A;
    const float *bf = (const float *)B;
    for (int64_t k0 = 0; k0 < K; k0 += k_blk) {
      int64_t kc = K - k0 < k_blk ? K - k0 : k_blk;
      __weft_onednn_brgemm_execute(
          af + k0, bf + k0 * ldb, C,
          M, N, kc, lda, ldb, ldc,
          dt_a, dt_b, dt_c, k0 == 0 ? add_c : 1);
      if (__weft_tls_status != 0) return;
    }
    return;
  }
  dnnl_data_type_t dnnl_a, dnnl_b, dnnl_c;
  if (__weft_map_dnnl_dt(dt_a, &dnnl_a) != 0 ||
      __weft_map_dnnl_dt(dt_b, &dnnl_b) != 0 ||
      __weft_map_dnnl_dt(dt_c, &dnnl_c) != 0) {
    __weft_record_error(1, "Weft oneDNN BRGeMM received invalid dtype code (%d/%d/%d)",
                        dt_a, dt_b, dt_c);
    return;
  }
#if defined(__linux__) && defined(__x86_64__)
  if (dt_a >= 2) {
    static _Thread_local int amx_ready = 0;
    if (!amx_ready) {
      unsigned long permitted = 0;
      if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &permitted) == 0 &&
          !(permitted & (1ul << 18))) {
        (void)syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18ul);
      }
      amx_ready = 1;
    }
  }
#endif
  enum { CACHE_SLOTS = 64 };
  static _Thread_local struct __weft_brgemm_entry cache[CACHE_SLOTS];
  static _Thread_local uint64_t clock_tick = 0;
  static _Thread_local void *scratch_ptr = NULL;
  static _Thread_local size_t scratch_cap = 0;

  uint64_t h = (uint64_t)M * 11400714819323198485ull;
  h ^= (uint64_t)N * 14029467366897019727ull;
  h ^= (uint64_t)K * 1609587929392839161ull;
  h ^= (uint64_t)lda * 9650029242287828579ull;
  h ^= (uint64_t)ldb * 2800177450012600261ull;
  h ^= (uint64_t)ldc * 6364136223846793005ull;
  h ^= (uint64_t)((dt_a & 15) | ((dt_b & 15) << 4) | ((dt_c & 15) << 8) | ((add_c & 1) << 12));

  struct __weft_brgemm_entry *hit = NULL;
  size_t base = (size_t)((h >> 32) ^ h) & (CACHE_SLOTS - 1);
  size_t victim = base;
  uint64_t oldest = UINT64_MAX;
  for (size_t probe = 0; probe < 8; ++probe) {
    size_t slot = (base + probe) & (CACHE_SLOTS - 1);
    struct __weft_brgemm_entry *e = &cache[slot];
    if (!e->valid) {
      victim = slot;
      oldest = 0;
      break;
    }
    if (e->M == M && e->N == N && e->K == K &&
        e->lda == lda && e->ldb == ldb && e->ldc == ldc &&
        e->dt_a == dt_a && e->dt_b == dt_b && e->dt_c == dt_c && e->add_c == add_c) {
      hit = e;
      break;
    }
    if (e->stamp < oldest) {
      oldest = e->stamp;
      victim = slot;
    }
  }

  if (!hit) {
    dnnl_brgemm_t brg = NULL;
    dnnl_status_t st = dnnl_brgemm_create(
        &brg, (dnnl_dim_t)M, (dnnl_dim_t)N, (dnnl_dim_t)K, 1,
        (dnnl_dim_t)lda, (dnnl_dim_t)ldb, (dnnl_dim_t)ldc,
        dnnl_a, dnnl_b, dnnl_c);
    if (st != dnnl_success || !brg) {
      if (brg) dnnl_brgemm_destroy(brg);
      __weft_record_error(
          __weft_status_from_dnnl(st, 5),
          "Weft oneDNN BRGeMM create failed (dnnl_status=%d, M=%lld N=%lld K=%lld lda=%lld ldb=%lld ldc=%lld dt=%d/%d/%d)",
          (int)st, (long long)M, (long long)N, (long long)K,
          (long long)lda, (long long)ldb, (long long)ldc, dt_a, dt_b, dt_c);
      return;
    }
    if (add_c) {
      st = dnnl_brgemm_set_add_C(brg, 1);
      if (st != dnnl_success) {
        dnnl_brgemm_destroy(brg);
        __weft_record_error(__weft_status_from_dnnl(st, 5),
                            "Weft oneDNN BRGeMM set_add_C failed (dnnl_status=%d)", (int)st);
        return;
      }
    }
    st = dnnl_brgemm_finalize(brg);
    if (st != dnnl_success) {
      dnnl_brgemm_destroy(brg);
      __weft_record_error(__weft_status_from_dnnl(st, 5),
                          "Weft oneDNN BRGeMM finalize failed (dnnl_status=%d, M=%lld N=%lld K=%lld)",
                          (int)st, (long long)M, (long long)N, (long long)K);
      return;
    }
    st = dnnl_brgemm_generate(brg);
    if (st != dnnl_success) {
      dnnl_brgemm_destroy(brg);
      __weft_record_error(__weft_status_from_dnnl(st, 4),
                          "Weft oneDNN BRGeMM generate failed (dnnl_status=%d, M=%lld N=%lld K=%lld)",
                          (int)st, (long long)M, (long long)N, (long long)K);
      return;
    }
    size_t sz = 0;
    st = dnnl_brgemm_get_scratchpad_size(brg, &sz);
    if (st != dnnl_success || sz > (size_t)PTRDIFF_MAX - 4096u) {
      dnnl_brgemm_destroy(brg);
      __weft_record_error(__weft_status_from_dnnl(st, 6),
                          "Weft oneDNN BRGeMM scratchpad query failed (dnnl_status=%d)", (int)st);
      return;
    }
    struct __weft_brgemm_entry *e = &cache[victim];
    if (e->valid && e->brg) {
      if (__weft_active_hw_brg == (const void *)e->brg) {
        weft_kernel_release_hw_context();
      }
      dnnl_brgemm_destroy(e->brg);
    }
    e->M = M; e->N = N; e->K = K;
    e->lda = lda; e->ldb = ldb; e->ldc = ldc;
    e->dt_a = dt_a; e->dt_b = dt_b; e->dt_c = dt_c; e->add_c = add_c;
    e->brg = brg;
    e->scratch_bytes = sz;
    e->valid = 1;
    hit = e;
  }
  hit->stamp = ++clock_tick;

  if (hit->scratch_bytes > scratch_cap) {
    size_t next_cap = ((hit->scratch_bytes + 4095u) / 4096u) * 4096u;
    void *next_buf = NULL;
    if (posix_memalign(&next_buf, 64, next_cap) != 0 || !next_buf) {
      __weft_record_error(6, "Weft oneDNN BRGeMM scratchpad allocation failed (%zu bytes)", next_cap);
      return;
    }
    free(scratch_ptr);
    scratch_ptr = next_buf;
    scratch_cap = next_cap;
  }

  dnnl_dim_t ab_offsets[2] = {0, 0};
  dnnl_status_t st = dnnl_success;
  if (__weft_active_hw_brg != (const void *)hit->brg) {
    st = dnnl_brgemm_set_hw_context(hit->brg);
    if (st != dnnl_success) {
      __weft_record_error(__weft_status_from_dnnl(st, 4),
                          "Weft oneDNN BRGeMM set_hw_context failed (dnnl_status=%d)", (int)st);
      return;
    }
    __weft_active_hw_brg = (const void *)hit->brg;
    if (dt_a >= 2) {
      __weft_amx_hw_active = 1;
    }
  }
  st = dnnl_brgemm_execute(hit->brg, A, B, ab_offsets, C, scratch_ptr);
  if (st != dnnl_success) {
    weft_kernel_release_hw_context();
    __weft_record_error(__weft_status_from_dnnl(st, 6),
                        "Weft oneDNN BRGeMM execute failed (dnnl_status=%d, M=%lld N=%lld K=%lld)",
                        (int)st, (long long)M, (long long)N, (long long)K);
    return;
  }
}
'''


def launch_source(abi):
    from .dispatch import dispatch_source
    symbol = abi["symbol"]
    if not symbol.isascii() or not symbol.isidentifier():
        raise ValueError("invalid native kernel symbol")
    values, declarations, fields = [], [], []
    for index, argument in enumerate(abi["arguments"]):
        name = argument["name"]
        if name.startswith("__triton_pid_"):
            values.append("pid" + name[-1])
        elif name.startswith("__triton_grid_"):
            values.append("grid" + name[-1])
        elif argument["kind"] in ("view", "pointer"):
            fields.append(f"{argument['c_type']} value_{index};")
            declarations.append(f"decoded.value_{index} = ({argument['c_type']})arguments[{index}];")
            values.append(f"decoded->value_{index}")
        else:
            fields.append(f"{argument['c_type']} value_{index};")
            declarations.append(f"memcpy(&decoded.value_{index}, arguments[{index}], sizeof(decoded.value_{index}));")
            values.append(f"decoded->value_{index}")
    count = len(abi["arguments"])
    extras = len(abi["shape_parameters"]) + sum(len(arg["shape"]) for arg in abi["arguments"] if arg["strided"])
    for index in range(count, count + extras):
        fields.append(f"size_t value_{index};")
        declarations.append(f"memcpy(&decoded.value_{index}, arguments[{index}], sizeof(decoded.value_{index}));")
        values.append(f"decoded->value_{index}")
    invocation = f"(({symbol}__weft_entry_type)entry)({', '.join(values)});"
    uses_brgemm = bool(abi.get("uses_onednn_brgemm"))
    reset_prefix = "__weft_reset_tls_status(); " if uses_brgemm else ""
    dispatch = dispatch_source(abi).replace('#include "kernel.h"', '')
    if uses_brgemm:
        dispatch = dispatch.replace(
            'void weft_kernel_dispatch(void (*entry)(void), void *const *arguments) {',
            'void weft_kernel_dispatch(void (*entry)(void), void *const *arguments) {\n    __weft_reset_tls_status();')
    single = single_instance_abi(abi)
    # Private wrapper parameter names must not hide the kernel symbol. Resolve
    # the target in file scope, just as the typed indirect dispatch does.
    parameters, single_args = [], []
    for arg in abi["arguments"]:
        if arg["name"].startswith("__triton_pid_"):
            single_args.append("0")
        elif arg["name"].startswith("__triton_grid_"):
            single_args.append("1")
        else:
            name = f"weft_arg_{len(parameters)}"
            parameters.append(f"{arg['c_type']} {name}")
            single_args.append(name)
    for i in range(extras):
        name = f"weft_arg_{len(parameters)}"
        parameters.append(f"size_t {name}")
        single_args.append(name)
    direct = symbol + "__weft_direct"
    single_body = (f"static {symbol}__weft_entry_type const {direct} = &{symbol};\n"
                   f"void {single['symbol']}({', '.join(parameters) or 'void'}) {{ "
                   + reset_prefix + direct + "(" + ", ".join(single_args) + "); }")
    instance = symbol + "__weft_launch_instance"
    single_dispatch = dispatch_source(single).replace('#include "kernel.h"', '').replace(
        'void weft_kernel_dispatch(', 'void triton_weft_instance_dispatch(')
    onednn_shim = _ONEDNN_BRGEMM_SHIM if uses_brgemm else _WEFT_STATUS_SHIM_NOOP
    return f'''#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include <stdatomic.h>
{onednn_shim}
{abi['declaration']}
{dispatch}
{single_body}
{single_dispatch}
{GRID_DECLARATION}
struct launch_arguments {{ {' '.join(fields) if fields else 'unsigned char unused;'} }};
typedef void (*triton_parallel_body)(void *, size_t, size_t);
typedef int (*triton_parallel_runner)(size_t, int, const void *, triton_check_thread,
                                      void *, triton_parallel_body, char *, size_t);
static _Atomic(triton_parallel_runner) parallel_executor = NULL;
void triton_weft_bind_parallel(triton_parallel_runner executor) {{
  atomic_store_explicit(&parallel_executor, executor, memory_order_release);
}}
static inline void {instance}(void (*entry)(void), const struct launch_arguments *decoded,
                            size_t grid0, size_t grid1, size_t grid2,
                            size_t pid0, size_t pid1, size_t pid2) {{
  {invocation}
}}
struct launch_job {{
  void (*entry)(void);
  const struct launch_arguments *decoded;
  size_t grid0, grid1, grid2;
  _Atomic(int) status;
  _Atomic(int) err_ready;
  char error[512];
}};
static void launch_range(void *opaque, size_t begin, size_t end) {{
  struct launch_job *job = (struct launch_job *)opaque;
  (void)weft_kernel_take_status(NULL, 0);
  size_t pid0 = begin % job->grid0;
  const size_t row = begin / job->grid0;
  size_t pid1 = row % job->grid1, pid2 = row / job->grid1;
  for (size_t i = begin; i < end; ++i) {{
    if (atomic_load_explicit(&job->status, memory_order_relaxed) != 0) break;
    {instance}(job->entry, job->decoded, job->grid0, job->grid1, job->grid2,
               pid0, pid1, pid2);
    char worker_err[512];
    int worker_st = weft_kernel_take_status(worker_err, sizeof(worker_err));
    if (worker_st != 0) {{
      int expected = 0;
      if (atomic_compare_exchange_strong_explicit(&job->status, &expected, worker_st,
                                                  memory_order_acq_rel, memory_order_relaxed)) {{
        strncpy(job->error, worker_err, sizeof(job->error) - 1);
        job->error[sizeof(job->error) - 1] = 0;
        atomic_store_explicit(&job->err_ready, 1, memory_order_release);
      }}
      break;
    }}
    if (++pid0 == job->grid0) {{
      pid0 = 0;
      if (++pid1 == job->grid1) {{ pid1 = 0; ++pid2; }}
    }}
  }}
  weft_kernel_release_hw_context();
}}
int triton_weft_grid(void (*entry)(void), void *const *arguments,
                    size_t grid0, size_t grid1, size_t grid2, int threads,
                    const void *scope, triton_check_thread check, char *error, size_t error_size) {{
  if (!grid0 || !grid1 || !grid2) return 0;
  if (grid0 > SIZE_MAX / grid1 || grid0 * grid1 > SIZE_MAX / grid2) return 1;
  size_t count = grid0 * grid1 * grid2;
  if (count > PTRDIFF_MAX || threads < 0 || !check) return 1;
  struct launch_arguments decoded;
  {' '.join(declarations)}
  if (threads == 0) threads = omp_get_max_threads();
  struct launch_job job = {{entry, &decoded, grid0, grid1, grid2, 0, 0, {{0}}}};
  if (count == 1 || threads == 1) {{
    int status = check(scope, error, error_size);
    if (status) return status;
    (void)weft_kernel_take_status(NULL, 0);
    if (count == 1) {{
      {instance}(entry, &decoded, grid0, grid1, grid2, 0, 0, 0);
      weft_kernel_release_hw_context();
      return weft_kernel_take_status(error, error_size);
    }}
    launch_range(&job, 0, count);
    int body_st = atomic_load_explicit(&job.status, memory_order_acquire);
    if (body_st != 0 && error && error_size &&
        atomic_load_explicit(&job.err_ready, memory_order_acquire)) {{
      strncpy(error, job.error, error_size - 1);
      error[error_size - 1] = 0;
    }}
    return body_st;
  }}
  triton_parallel_runner executor = atomic_load_explicit(&parallel_executor, memory_order_acquire);
  if (!executor) {{
    if (error && error_size) {{
      strncpy(error, "native grid needs the shared Weft parallel runtime", error_size - 1);
      error[error_size - 1] = 0;
    }}
    return 5;
  }}
  int exec_st = executor(count, threads, scope, check, &job, launch_range, error, error_size);
  if (exec_st != 0) return exec_st;
  int body_st = atomic_load_explicit(&job.status, memory_order_acquire);
  if (body_st != 0 && error && error_size) {{
    while (!atomic_load_explicit(&job.err_ready, memory_order_acquire)) {{}}
    strncpy(error, job.error, error_size - 1);
    error[error_size - 1] = 0;
  }}
  return body_st;
}}
'''
