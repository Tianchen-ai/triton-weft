from __future__ import annotations

from triton.integrations import ggml_formats
from ..blocks.quantize import q8_k as quant_q8_k
import triton
import triton.language as tl

from ..vec_dot.grouped_lookup import (
    compute_iq1_m, compute_iq1_s, compute_iq2_xs, compute_iq2_xxs,
)
from ..blocks.vec_dot.iq2_xxs_q8_k import compute as compute_iq2_xxs_tensor
from ..blocks.vec_dot.iq3_xxs_q8_k import compute as compute_iq3_xxs_tensor
from ..vec_dot.iq1 import compute_s_windowed


@triton.jit(noinline=True)
def row_iq1_m(W: tl.View[ggml_formats.IQ1_M, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
              grid: tl.View[ggml_formats.I8X8, (2048, 8)], DEFERRED: tl.constexpr):
    return compute_iq1_m(W, X, grid, K, DEFERRED)


@triton.jit(noinline=True)
def row_iq1_s(W: tl.View[ggml_formats.IQ1_S, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
              grid: tl.View[ggml_formats.I8X8, (2048, 8)], GROUPS: tl.constexpr):
    if GROUPS:
        return compute_s_windowed(W, X, grid, K, GROUPS)
    return compute_iq1_s(W, X, grid, K)


@triton.jit(noinline=True)
def row_iq2_xs(W: tl.View[ggml_formats.IQ2_XS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
               grid: tl.View[ggml_formats.I8X8, (512, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)]):
    return compute_iq2_xs(W, X, grid, signs, K)


@triton.jit(noinline=True)
def row_iq2_xxs(W: tl.View[ggml_formats.IQ2_XXS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                grid: tl.View[ggml_formats.I8X8, (256, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)],
                PAIR_ACCUM: tl.constexpr):
    return compute_iq2_xxs(W, X, grid, signs, K, PAIR_ACCUM)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq1_m(W: tl.View[ggml_formats.IQ1_M, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
           Y: tl.View[tl.f32, (M, N)], *, DEFERRED: tl.constexpr = False):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq1_m(W[column], Xq[row], grid, DEFERRED))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
           Y: tl.View[tl.f32, (M, N)], *, GROUPS: tl.constexpr = 0,
           INLINE: tl.constexpr = False):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            if INLINE:
                if GROUPS:
                    value = compute_s_windowed(W[column], Xq[row], grid, K, GROUPS)
                else:
                    value = compute_iq1_s(W[column], Xq[row], grid, K)
            else:
                value = row_iq1_s(W[column], Xq[row], grid, GROUPS)
            tl.store(Y[row, column], value)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xs(W: tl.View[ggml_formats.IQ2_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (512, 8)],
            signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq2_xs(W[column], Xq[row], grid, signs))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xxs(W: tl.View[ggml_formats.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
             Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (256, 8)],
             signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)],
             *, INLINE: tl.constexpr = False, PAIR_ACCUM: tl.constexpr = False):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            if INLINE:
                value = compute_iq2_xxs(W[column], Xq[row], grid, signs, K, PAIR_ACCUM)
            else:
                value = row_iq2_xxs(W[column], Xq[row], grid, signs, PAIR_ACCUM)
            tl.store(Y[row, column], value)


@triton.jit(noinline=True)
def row_iq2_xxs_tensor(W: tl.View[ggml_formats.IQ2_XXS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                      grid: tl.View[ggml_formats.I8X8, (256, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)]):
    return compute_iq2_xxs_tensor(W, X, grid, signs)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xxs_tensor(W: tl.View[ggml_formats.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
                   Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (256, 8)],
                   signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq2_xxs_tensor(W[column], Xq[row], grid, signs))


@triton.jit(noinline=True)
def row_iq3_xxs_tensor(W: tl.View[ggml_formats.IQ3_XXS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                      grid: tl.View[ggml_formats.I8X4, (256, 4)], signs: tl.View[ggml_formats.I8X8, (128, 8)],
                      INDEX_BITS: tl.constexpr):
    return compute_iq3_xxs_tensor(W, X, grid, signs, INDEX_BITS)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq3_xxs_tensor(W: tl.View[ggml_formats.IQ3_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
                   Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X4, (256, 4)],
                   signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)],
                   *, INDEX_BITS: tl.constexpr = 32):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq3_xxs_tensor(W[column], Xq[row], grid, signs, INDEX_BITS))
