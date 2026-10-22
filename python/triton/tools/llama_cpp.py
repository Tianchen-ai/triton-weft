from __future__ import annotations

import re
from pathlib import Path

from triton.runtime.weft.errors import CompilationError
from triton.runtime.weft.native import NativeSignature


def _interface(kernel):
    abi = kernel.metadata.single_instance_abi
    return NativeSignature(abi, bindings=abi["bindings"])


# Framework storage contracts, not lowering rules or alternative math kernels.
# The same facts are checked against the actual linked GGML library on load.
VEC_DOT = {
    "q4_0": ("Q4_0", "Q8_0", 32, 18, 34),
    "q4_1": ("Q4_1", "Q8_1", 32, 20, 36),
    "q5_0": ("Q5_0", "Q8_0", 32, 22, 34),
    "q5_1": ("Q5_1", "Q8_1", 32, 24, 36),
    "q8_0": ("Q8_0", "Q8_0", 32, 34, 34),
    "q2_K": ("Q2_K", "Q8_K", 256, 84, 292),
    "q3_K": ("Q3_K", "Q8_K", 256, 110, 292),
    "q4_K": ("Q4_K", "Q8_K", 256, 144, 292),
    "q5_K": ("Q5_K", "Q8_K", 256, 176, 292),
    "q6_K": ("Q6_K", "Q8_K", 256, 210, 292),
}


def validate_vec_dot(artifact, weight_type):
    if weight_type not in VEC_DOT:
        raise NotImplementedError(f"GGML adapter has no declared storage contract for {weight_type}")
    weight, activation, block, weight_bytes, activation_bytes = VEC_DOT[weight_type]
    interface = _interface(artifact)
    parameters = interface.arguments
    if len(parameters) != 3 or interface.shape_parameters != ("K",):
        raise ValueError("GGML vec-dot requires exactly W, X, Y Views and one dynamic K")
    for parameter, encoding, shape, access, elements, size in zip(
            parameters, (weight, activation, "dense.f32"), (("K",), ("K",), ("1",)),
            ("read", "read", "write"), (block, block, 1), (weight_bytes, activation_bytes, 4)):
        if (parameter.kind, parameter.encoding, parameter.shape, parameter.access,
            parameter.record_elements, parameter.storage_bytes, parameter.strided, parameter.interleave_rows) != (
                "view", encoding, shape, access, elements, size, False, 0):
            raise ValueError(f"{parameter.name} does not match the GGML {weight_type} vec-dot ABI")
    if artifact.metadata.single_instance_abi["used_extensions"]:
        raise ValueError("GGML vec-dot registration only accepts ordinary RVV entries; matrix paths have different contracts")


def validate_quantizer(artifact, activation_type):
    contracts = {"q8_0": ("Q8_0", 32, 34, 2), "q8_K": ("Q8_K", 256, 292, 4)}
    if activation_type not in contracts:
        raise NotImplementedError(f"GGML activation quantizer has no storage contract for {activation_type}")
    encoding, block, size, alignment = contracts[activation_type]
    interface = _interface(artifact)
    arguments = interface.arguments
    if len(arguments) != 2 or interface.shape_parameters != ("K",):
        raise ValueError("GGML activation quantization requires X[K], Y[K] and one runtime K")
    for arg, enc, access, elements, storage, align in zip(
            arguments, ("dense.f32", encoding), ("read", "write"), (1, block), (4, size), (4, alignment)):
        if (arg.kind, arg.encoding, arg.shape, arg.access, arg.record_elements,
                arg.storage_bytes, arg.strided, arg.interleave_rows) != (
                "view", enc, ("K",), access, elements, storage, False, 0) or arg.alignment > align:
            raise ValueError(f"{arg.name} does not match the GGML activation row ABI")
    if artifact.metadata.single_instance_abi["used_extensions"]:
        raise ValueError("GGML activation quantization requires an ordinary CPU row entry")


def validate_ime1(artifact, role):
    contracts = {"q4_centered_prefill": (False, 4), "q4_affine_prefill": (True, 4),
                 "q4_centered_decode": (False, 1), "q4_affine_decode": (True, 1)}
    if role not in contracts:
        raise NotImplementedError(f"GGML IME row tile has no contract for {role}")
    affine, rows = contracts[role]
    interface = _interface(artifact)
    arguments = interface.arguments
    if len(arguments) != 3 or interface.shape_parameters != ("N", "K", "M"):
        raise ValueError("GGML IME row tile requires W, Xq, Y and N, K, M dimensions")
    encodings = ("IMEQ4Affine_I[16]" if affine else "IMEQ4Centered_I[16]", f"IMEQ8_I[{rows}]", "dense.f32")
    for i, arg in enumerate(arguments):
        if (arg.kind, arg.encoding, arg.shape, arg.access, arg.record_elements,
                arg.storage_bytes, arg.strided, arg.interleave_rows) != (
                "view", encodings[i], (("N", "K"), ("M", "K"), ("M", "N"))[i],
                "write" if i == 2 else "read", 1 if i == 2 else 32,
                ((19 if affine else 18), 36, 4)[i], i == 2, (16, rows, 0)[i]) or arg.alignment > ((1 if affine else 2), 4, 4)[i]:
            raise ValueError(f"{arg.name} does not match the GGML IME row-tile ABI")
    if tuple(artifact.metadata.single_instance_abi["used_extensions"]) != ("spacemit-ime1",):
        raise ValueError("GGML IME binding requires an artifact using the IME1 capability")


def validate_ime2(artifact, role):
    if role not in ("i8_decode", "i8_prefill"):
        raise NotImplementedError(f"GGML IME2 has no packed numerical contract for {role}")
    rows = 1 if role == "i8_decode" else 4
    interface = _interface(artifact)
    if len(interface.arguments) != 3 or interface.shape_parameters != ("N", "K", "M"):
        raise ValueError("GGML IME2 row tile requires W, Xq, Y and N, K, M dimensions")
    encodings = ("IME2Weight8Rows[32]", f"IME2Activation8Rows[{rows}]", "dense.f32")
    for i, arg in enumerate(interface.arguments):
        if (arg.kind, arg.encoding, arg.shape, arg.access, arg.record_elements,
                arg.storage_bytes, arg.strided, arg.interleave_rows) != (
                "view", encodings[i], (("N", "K"), ("M", "K"), ("M", "N"))[i],
                "write" if i == 2 else "read", 1 if i == 2 else 32,
                (34, 38, 4)[i], i == 2, (32, rows, 0)[i]) or arg.alignment > (2, 2, 4)[i]:
            raise ValueError(f"{arg.name} does not match the native IME2 i8 block32 ABI")
    if tuple(artifact.metadata.single_instance_abi["used_extensions"]) != ("spacemit-ime2",):
        raise ValueError("GGML IME2 binding requires a selected IME2 instruction")


def validate_amx(artifact, role):
    if role not in ("q4_0_decode", "q4_0_prefill", "q4_K_decode", "q4_K_prefill", "q6_K_decode", "q6_K_prefill"):
        raise NotImplementedError(f"GGML AMX buffer has no contract for {role}")
    prefill = role.endswith("_prefill")
    k_block = role in ("q4_K_decode", "q4_K_prefill", "q6_K_decode", "q6_K_prefill")
    weight, activation, block, weight_bytes, activation_bytes, activation_alignment = (
        ("AMXQ6_KRows[16]", "Q8_K", 256, 210, 292, 4) if role in ("q6_K_decode", "q6_K_prefill") else
        ("AMXQ4_KRows[16]", "Q8_K", 256, 148, 292, 4) if k_block else
        ("AMXQ4_0Rows[16]", "Q8_0", 32, 18, 34, 2))
    interface = _interface(artifact)
    fixed = prefill and interface.shape_parameters == ("K",)
    dimensions = ("K",) if fixed else ("N", "K", "M") if prefill else ("N", "K")
    scratch = not prefill and not k_block and len(interface.arguments) == 4
    matrix_scratch = prefill and not k_block and not fixed and len(interface.arguments) == 4
    if (len(interface.arguments) != 3 and not scratch and not matrix_scratch) or interface.shape_parameters != dimensions:
        raise ValueError(f"GGML packed row tile requires W, Xq, Y and {dimensions} dimensions")
    shapes = ((("32", "K"), ("32", "K"), ("32", "32")) if fixed else
              (("N", "K"), ("M", "K"), ("M", "N")) if prefill else (("N", "K"), ("K",), ("N",)))
    for index, (arg, encoding, shape, access, elements, size, rows, alignment) in enumerate(zip(
            interface.arguments, (weight, activation, "dense.f32"),
            shapes, ("read", "read", "write"),
            (block, block, 1), (weight_bytes, activation_bytes, 4), (16, 0, 0), (2, activation_alignment, 4))):
        if (arg.kind, arg.encoding, arg.shape, arg.access, arg.record_elements,
                arg.storage_bytes, arg.strided, arg.interleave_rows) != (
                "view", encoding, shape, access, elements, size, prefill and index == 2, rows) or arg.alignment > alignment:
            raise ValueError(f"{arg.name} does not match the GGML AMX {role} row-tile layout")
    if scratch:
        arg = interface.arguments[3]
        if (arg.kind, arg.encoding, arg.shape, arg.access, arg.record_elements,
                arg.storage_bytes, arg.strided, arg.interleave_rows) != (
                "view", "BlockSum32", ("K",), "readwrite", 32, 2, False, 0) or arg.alignment > 2:
            raise ValueError("AMX decode scratch requires signed sums for each complete Q8_0 record")
    if matrix_scratch:
        arg = interface.arguments[3]
        if (arg.kind, arg.encoding, arg.shape, arg.access, arg.record_elements,
                arg.storage_bytes, arg.strided, arg.interleave_rows) != (
                "view", "BlockScale32Rows[32]", ("M", "K"), "readwrite", 32, 4, False, 32) or arg.alignment > 4:
            raise ValueError("AMX prefill scratch requires F32 scales interleaved across 32 rows")
    abi = artifact.metadata.single_instance_abi
    allowed = {"amx-int8"} if prefill else set()
    if abi["arch"] != "x86_64" or set(abi["used_extensions"]) - allowed:
        raise ValueError("packed row tiles require x86 SIMD, with optional integer AMX for prefill")


HOOK_HEADER = '''/* Weft CPU call-site integration. Mathematics stays in generated artifacts. */
#pragma once
#if defined(GGML_USE_WEFT)
#include <weft/ggml.h>
#define WEFT_GGML_INIT() ggml_weft_initialize()
#define WEFT_GGML_DOT_TYPE ggml_weft_dot_binding
#define WEFT_GGML_DOT(type, original) ggml_weft_select_vec_dot((type), (original))
#define WEFT_GGML_CALL_DOT(binding, ...) ggml_weft_call_dot((binding), __VA_ARGS__)
#define WEFT_GGML_FROM_FLOAT(fn, x, y, n) ggml_weft_from_float((fn), (x), (y), (n))
#define WEFT_GGML_NODE(params, tensor) ggml_weft_begin_node((tensor), (params)->ith, (params)->use_ref)
#define WEFT_GGML_GRAPH(graph, plan) ggml_weft_begin_graph((graph)->n_nodes, (graph)->nodes, (plan)->abort_callback != NULL)
#define WEFT_GGML_END_NODE() ggml_weft_end_node()
#define WEFT_GGML_EXTRA(params, tensor) ggml_weft_extra_node((tensor), (params)->ith)
#else
#define WEFT_GGML_INIT() ((void)0)
#define WEFT_GGML_DOT_TYPE ggml_vec_dot_t
#define WEFT_GGML_DOT(type, original) (original)
#define WEFT_GGML_CALL_DOT(binding, ...) (binding)(__VA_ARGS__)
#define WEFT_GGML_FROM_FLOAT(fn, x, y, n) (fn)((x), (y), (n))
#define WEFT_GGML_NODE(params, tensor) ((void)0)
#define WEFT_GGML_GRAPH(graph, plan) ((void)0)
#define WEFT_GGML_END_NODE() ((void)0)
#define WEFT_GGML_EXTRA(params, tensor) ((void)0)
#endif
'''


# The reviewed upstream vendor source uses C++20 and has unconditional source
# entries whose files reject absent ISA macros. Apply its own probe results to
# those entries in both ordinary and Weft builds; retain every available engine.
NATIVE_BUILD = '''
    if (GGML_CPU_RISCV64_SPACEMIT)
        target_compile_features(${GGML_CPU_NAME} PRIVATE cxx_std_20)
        set_property(TARGET ${GGML_CPU_NAME} PROPERTY CXX_SCAN_FOR_MODULES OFF)
        get_target_property(native_cpu_sources ${GGML_CPU_NAME} SOURCES)
        if (NOT SPACEMIT_RISCV_COMPILER_SUPPORT_IME1)
            list(FILTER native_cpu_sources EXCLUDE REGEX "/ime1_kernels[.]cpp$")
        endif()
        if (NOT (SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S4 AND SPACEMIT_RISCV_COMPILER_SUPPORT_VPACK AND SPACEMIT_RISCV_COMPILER_SUPPORT_VNPACK))
            list(FILTER native_cpu_sources EXCLUDE REGEX "/ime2_kernels[.]cpp$")
        endif()
        set_property(TARGET ${GGML_CPU_NAME} PROPERTY SOURCES "${native_cpu_sources}")
    endif()'''


def _once(source, before, after):
    if source.count(before) != 1:
        raise CompilationError(f"llama.cpp CPU source does not expose the supported integration boundary: {before!r}")
    return source.replace(before, after, 1)


def _spacemit_edits(root):
    directory = root / "ggml/src/ggml-spacemit"
    if not directory.is_dir():
        return {}
    path = directory / "ime.cpp"
    graph_path = directory / "ggml-spacemit.cpp"
    cmake_path = directory / "CMakeLists.txt"
    source, graph, build = (p.read_text(encoding="utf-8") for p in (path, graph_path, cmake_path))
    header = '#if defined(GGML_USE_WEFT)\n#include <weft/ggml.h>\n#endif\n'
    anchor = '                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;'
    selected = anchor + '''
#if defined(GGML_USE_WEFT)
                if (auto *binding = ggml_weft_select_ime2(spacemit_kernels::ime2::gemm_kernel_i8i8, src0->type)) {
                    gemm_kernel = [binding](size_t block, const uint8_t *x,
                            const uint8_t *w, const uint8_t *zero, float *y, size_t rows,
                            size_t columns, size_t blocks, size_t stride) {
                        return ggml_weft_call_ime2(binding, block, x, w, zero, y, rows, columns, blocks, stride);
                    };
                }
#endif'''
    graph_anchor = '            for (int i = 0; i < n_nodes; i++) {'
    begin = '''#if defined(GGML_USE_WEFT)
            ggml_weft_begin_graph(n_nodes, nodes, false);
#endif
'''
    node_anchor = '                if (!ggml_spacemit_compute_forward(ctx, node)) {'
    node_begin = '''#if defined(GGML_USE_WEFT)
                ggml_weft_begin_node(node, (int)ith, false);
#endif
'''
    node_end = '''#if defined(GGML_USE_WEFT)
                ggml_weft_end_node();
#endif
'''
    end_anchor = '                ggml_profile_log_spacemit_op_end(node, (int) ith, (int) runtime->grid_dim(0));'
    build_hook = '''
if (GGML_WEFT)
    # Share the CPU adapter/package and its counters across both backends.
    target_link_libraries(${TARGET_NAME} PRIVATE ggml-cpu)
    target_compile_definitions(${TARGET_NAME} PRIVATE GGML_USE_WEFT)
endif()
'''
    if "ggml_weft_select_ime2" in source:
        if source.count(selected) != 2 or begin not in graph or node_begin not in graph or node_end not in graph or build_hook not in build:
            raise CompilationError("llama.cpp contains an incomplete standalone SpacemiT integration")
        return {}
    if source.count(anchor) != 2:
        raise CompilationError("SpacemiT IME2 MUL_MAT and MUL_MAT_ID i8 boundaries have changed")
    source = header + source.replace(anchor, selected)
    graph = header + _once(graph, graph_anchor, begin + graph_anchor)
    graph = _once(graph, node_anchor, node_begin + node_anchor)
    graph = _once(graph, end_anchor, node_end + end_anchor)
    return {path: source, graph_path: graph, cmake_path: build + build_hook}


def install_spacemit_source(source_directory):
    """Install the standalone SpacemiT backend's native package call sites."""
    root = Path(source_directory).expanduser().resolve(strict=True)
    if not (root / "ggml/src/ggml-spacemit").is_dir():
        raise CompilationError("llama.cpp has no standalone SpacemiT backend")
    edits = _spacemit_edits(root)
    for path, content in edits.items():
        path.write_text(content, encoding="utf-8")
    return {"source": str(root), "changed": bool(edits), "files": [str(p) for p in edits]}


def install_source(source_directory):
    """Add thin build/call-site hooks to an explicitly selected upstream tree.

    This does not disable repack, OpenMP, IME, BLAS or any other upstream backend.
    All expected boundaries are checked before the first write. An upstream
    layout change is unsupported until reviewed, rather than guessed around.
    """
    root = Path(source_directory).expanduser().resolve(strict=True)
    spacemit_edits = _spacemit_edits(root)
    directory = root / "ggml/src/ggml-cpu"
    path, cmake = directory / "ggml-cpu.c", directory / "CMakeLists.txt"
    source, build = path.read_text(encoding="utf-8"), cmake.read_text(encoding="utf-8")
    ime_path = directory / "spacemit/ime.cpp"
    ime = ime_path.read_text(encoding="utf-8")
    amx_path = directory / "amx/mmq.cpp"
    amx = amx_path.read_text(encoding="utf-8")
    amx_header = '#if defined(GGML_USE_WEFT)\n#include <weft/ggml.h>\n#endif\n'
    amx_decode = '    if (M == 1) {\n'
    amx_decode_parallel = '\n        parallel_for_ggml(params, n_batch * NB, [&](int begin, int end) {'
    amx_call = '''#if defined(GGML_USE_WEFT)
        if (!params->use_ref && (TYPE == GGML_TYPE_Q4_0 || TYPE == GGML_TYPE_Q4_K || TYPE == GGML_TYPE_Q6_K)) {
            const auto weft_amx = ggml_weft_select_amx(TYPE, false, K);
            if (weft_amx.function || weft_amx.scratch_function) {
                parallel_for_ggml(params, n_batch * NB, [&](int begin, int end) {
                    const int KB = K / (TYPE == GGML_TYPE_Q4_0 ? QK4_0 : QK_K);
                    const int TILE_SIZE = TYPE == GGML_TYPE_Q4_0 ? get_tile_size<block_q4_0>() :
                                          TYPE == GGML_TYPE_Q4_K ? get_tile_size<block_q4_K>() : get_tile_size<block_q6_K>();
                    const int row_size_A = KB * (TYPE == GGML_TYPE_Q4_0 ? sizeof(block_q8_0) : sizeof(block_q8_K));
                    for (int first = begin; first < end;) {
                        const int batch_idx = first / NB;
                        const int nb = first % NB;
                        const int nb_start = nb * BLOCK_N;
                        const int group_end = std::min(end, (batch_idx + 1) * NB);
                        const int group_size = std::min(N - nb_start, (group_end - first) * BLOCK_N);
                        const int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                        const int64_t dst_offset = ggml_batch_offset(dst, batch_idx, ne2);
                        const char *wdata_batch = (const char *)wdata + batch_idx * row_size_A;
                        ggml_weft_call_amx(weft_amx,
                            (const uint8_t *)src0->data + src0_offset + PACKED_INDEX(nb * kTilesN, 0, KB, TILE_SIZE),
                            (const uint8_t *)wdata_batch, (float *)dst->data + dst_offset + nb_start, group_size, K);
                        first = group_end;
                    }
                });
                return;
            }
        }
#endif
'''
    amx_prefill_parallel = '\n    parallel_for_ggml(params, n_batch * MB * NB, [&](int begin, int end) {'
    amx_prefill_group = 'const int group_end = std::min(end, (i / NB + 1) * NB);'
    amx_dynamic_prefill = 'if (weft_amx.matrix_function || weft_amx.scratch_matrix) {'
    amx_prefill_types = 'if (!params->use_ref && (TYPE == GGML_TYPE_Q4_0 || TYPE == GGML_TYPE_Q4_K || TYPE == GGML_TYPE_Q6_K)) {'
    amx_invalidate = 'if (weft_amx.tile_state) ggml_tile_config_initialized = false;'
    amx_matrix = '                tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>('
    amx_prefill_call = '''                ggml_weft_call_amx_prefill(weft_amx,
                    (const uint8_t *)src0->data + src0_offset + PACKED_INDEX(nb * 2, 0, KB, TILE_SIZE),
                    (const uint8_t *)wdata_batch + mb_start * row_size_A,
                    (float *)dst->data + dst_offset + mb_start * N + nb_start, nb_size, K, mb_size, ldc);'''
    ime_header = '#if defined(GGML_USE_WEFT)\n#include <weft/ggml.h>\n#endif\n'
    ime_anchor = '                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;'
    ime_call = ime_anchor + '''
#if defined(GGML_USE_WEFT)
                if (!params->use_ref) {
                    auto *binding = ggml_weft_select_ime1(spacemit_kernels::ime1::gemm_kernel_i8i4, src0->type);
                    if (binding) gemm_kernel = [binding](size_t block, const uint8_t *x,
                            const uint8_t *w, const uint8_t *zero, float *y, size_t rows,
                            size_t columns, size_t blocks, size_t stride) {
                        return ggml_weft_call_ime1(binding, block, x, w, zero, y, rows, columns, blocks, stride);
                    };
                }
#endif'''
    marker = '#include "weft-ggml-hooks.h"'
    header = directory / "weft-ggml-hooks.h"
    spert_graph = 'static void ggml_graph_compute_spert_kernel(int ith, int nth, void * user_data) {'
    graph_paths = 2 if spert_graph in source else 1
    if marker in source:
        if (header.read_text(encoding="utf-8") != HOOK_HEADER or "weft_add_llama_backend" not in build or
                NATIVE_BUILD not in build or ime_call not in ime or
                amx_call not in amx or amx_prefill_call not in amx or amx_prefill_group not in amx or
                amx_dynamic_prefill not in amx or amx_prefill_types not in amx or amx_invalidate not in amx or
                amx.count("ggml_weft_select_amx(TYPE,") != 2 or amx.count("ggml_weft_from_float_packed(") != 2 or
                source.count('WEFT_GGML_END_NODE();') != graph_paths or
                source.count('WEFT_GGML_GRAPH(cgraph, cplan);') != graph_paths):
            raise CompilationError("llama.cpp contains an incomplete Weft integration")
        for p, content in spacemit_edits.items():
            p.write_text(content, encoding="utf-8")
        return {"source": str(root), "changed": bool(spacemit_edits),
                "files": [str(p) for p in spacemit_edits]}
    source = _once(source, '#include "ggml-cpu-impl.h"', '#include "ggml-cpu-impl.h"\n' + marker)
    if ime.count(ime_anchor) != 2:
        raise CompilationError("llama.cpp IME MUL_MAT and MUL_MAT_ID row-tile boundaries have changed")
    ime = ime_header + ime.replace(ime_anchor, ime_call)
    decode_begin = amx.index(amx_decode)
    parallel_begin = amx.index(amx_decode_parallel, decode_begin)
    amx = amx_header + amx[:parallel_begin] + amx_call + amx[parallel_begin:]
    # Native GGML caches its tile configuration per worker. A selected artifact
    # owns and releases its own tile state; invalidate only that worker's cached
    # readiness after the replaced traversal, before a later native node uses it.
    amx = _once(amx, 'inline void ggml_tile_config_init(void) {\n    static thread_local bool done = false;\n\n    if (done) {',
                'static thread_local bool ggml_tile_config_initialized = false;\n\n'
                'inline void ggml_tile_config_init(void) {\n    if (ggml_tile_config_initialized) {')
    amx = _once(amx, '    _tile_loadconfig(&tc);\n    done = true;',
                '    _tile_loadconfig(&tc);\n    ggml_tile_config_initialized = true;')
    # Select a complete framework traversal before entering the original native
    # dispatch. Its lambdas then retain their upstream captures and bodies;
    # other quantized types do not inherit an optional artifact's live binding.
    prefill_begin = amx.index('            for (int i = begin; i < end; ++i) {', amx.index(amx_prefill_parallel))
    prefill_end = amx.index('\n        });', prefill_begin)
    native_loop = amx[prefill_begin:prefill_end]
    call_begin = native_loop.index(amx_matrix)
    call_end = native_loop.index(');', call_begin) + 2
    selected_loop = _once(native_loop, native_loop[call_begin:call_end], amx_prefill_call)
    grouped_loop = _once(selected_loop, 'for (int i = begin; i < end; ++i)',
                         'for (int i = begin; i < end;)')
    grouped_loop = _once(grouped_loop, 'int nb_size = BLOCK_N;',
                        amx_prefill_group + '\n'
                        '                int nb_size = std::min(N - nb_start, (group_end - i) * BLOCK_N);')
    grouped_loop = _once(grouped_loop, amx_prefill_call,
                        amx_prefill_call + '\n                i = group_end;')
    # A dynamic matrix artifact owns its column traversal. Coalesce only the
    # adjacent columns already assigned to this worker and input-row block.
    # Fixed 32-by-32 artifacts retain their exact invocation geometry.
    dispatch = '''#if defined(GGML_USE_WEFT)
    ''' + amx_prefill_types + '''
        const auto weft_amx = ggml_weft_select_amx(TYPE, true, K);
        if (weft_amx.matrix_function || ((weft_amx.scratch_matrix || weft_amx.fixed_matrix) && M % BLOCK_M == 0)) {
            parallel_for_ggml(params, n_batch * MB * NB, [&](int begin, int end) {
                const int KB = K / (TYPE == GGML_TYPE_Q4_0 ? QK4_0 : QK_K);
                const int TILE_SIZE = TYPE == GGML_TYPE_Q4_0 ? get_tile_size<block_q4_0>() :
                                      TYPE == GGML_TYPE_Q4_K ? get_tile_size<block_q4_K>() : get_tile_size<block_q6_K>();
                const int row_size_A = KB * (TYPE == GGML_TYPE_Q4_0 ? sizeof(block_q8_0) : sizeof(block_q8_K));
                ''' + amx_dynamic_prefill + '\n' + grouped_loop + '''
                } else {
''' + selected_loop + '''
                }
            });
            ''' + amx_invalidate + '''
            return;
        }
    }
#endif
'''
    amx = _once(amx, amx_prefill_parallel, dispatch + amx_prefill_parallel)
    # AMX prepares the ordinary Q8 records inside its extra-buffer path. Keep
    # its actual original quantizer (Q8_K uses the reference implementation)
    # when no artifact is bound; changing to the CPU trait here would change
    # the upstream numerical program even in the unbound control.
    for activation, function in (("q8_0", "quantize_row_q8_0"), ("q8_K", "quantize_row_q8_K_ref")):
        original = f"    {function}(x, (block_{activation} *)vy, k);"
        selected = f'''#if defined(GGML_USE_WEFT)
    ggml_weft_from_float_packed(GGML_TYPE_{activation.upper()},
        [](const float *source, void *destination, int64_t count) {{
            {function}(source, static_cast<block_{activation} *>(destination), count);
        }}, x, vy, k);
#else
{original}
#endif'''
        amx = _once(amx, original, selected)
    source, count = re.subn(r'ggml_vec_dot_t(\s+const vec_dot\s*=\s*)(type_traits_cpu\[type\]\.vec_dot);',
                            r'WEFT_GGML_DOT_TYPE\1WEFT_GGML_DOT(type, \2);', source)
    if count != 2:
        raise CompilationError("llama.cpp needs the reviewed MUL_MAT and MUL_MAT_ID vec-dot call sites")
    source, count = re.subn(r'\bfrom_float\(', 'WEFT_GGML_FROM_FLOAT(from_float, ', source)
    if count != 4:
        raise CompilationError("llama.cpp activation quantization boundaries have changed")
    source = _once(source, '    // extra_buffer op?\n', '    WEFT_GGML_NODE(params, tensor);\n\n    // extra_buffer op?\n')
    source, count = re.subn(r'(?m)^(\s*)vec_dot\(', r'\1WEFT_GGML_CALL_DOT(vec_dot, ', source)
    if count != 2:
        raise CompilationError("llama.cpp ordinary and indirect vec-dot invocation boundaries have changed")
    source = _once(source, '    if (ggml_cpu_extra_compute_forward(params, tensor)) {\n',
                   '    if (ggml_cpu_extra_compute_forward(params, tensor)) {\n        WEFT_GGML_EXTRA(params, tensor);\n')
    source = _once(source, '        is_first_call = false;\n', '        WEFT_GGML_INIT();\n        is_first_call = false;\n')
    # Newer SpacemiT trees also execute the same graph inside an AI-core
    # callback. Each worker path needs its own permission scope and node-end
    # accounting; instrumenting only the pthread loop would miss that path.
    spert_loop = '    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {'
    forward = '            ggml_compute_forward(&params, node);'
    if source.count(forward) != graph_paths:
        raise CompilationError("llama.cpp graph worker invocation boundaries have changed")
    source = source.replace(forward, forward + '\n            WEFT_GGML_END_NODE();')
    if graph_paths == 2:
        source = _once(source, spert_loop, '    WEFT_GGML_GRAPH(cgraph, cplan);\n\n' + spert_loop)
    graph_loop = '    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {'
    source = _once(source, graph_loop, '    WEFT_GGML_GRAPH(cgraph, cplan);\n\n' + graph_loop)
    build = 'option(GGML_WEFT "Use an explicit Weft kernel package at CPU call sites" OFF)\n\n' + build
    anchor = '    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})'
    build = _once(build, anchor, anchor + NATIVE_BUILD + '''
    if (GGML_WEFT)
        find_package(Weft CONFIG REQUIRED)
        include("${Weft_LLAMA_CPP_DIR}/WeftLlama.cmake")
        weft_add_llama_backend(${GGML_CPU_NAME})
    endif()''')
    if header.exists():
        raise FileExistsError(header)
    header.write_text(HOOK_HEADER, encoding="utf-8")
    path.write_text(source, encoding="utf-8")
    cmake.write_text(build, encoding="utf-8")
    ime_path.write_text(ime, encoding="utf-8")
    amx_path.write_text(amx, encoding="utf-8")
    for p, content in spacemit_edits.items():
        p.write_text(content, encoding="utf-8")
    return {"source": str(root), "changed": True,
            "files": [str(p) for p in (path, cmake, header, ime_path, amx_path, *spacemit_edits)]}
