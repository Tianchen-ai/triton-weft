from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import triton
import triton.language as tl

from weft.ggml.vec_dot.grouped_lookup import (
    compute_iq1_m, compute_iq1_s, compute_iq2_xs, compute_iq2_xxs,
)
from kernels.vec_dot.iq2_xxs_q8_k import compute as compute_iq2_xxs_tensor
from kernels.vec_dot.iq3_xxs_q8_k import compute as compute_iq3_xxs_tensor


@triton.jit(noinline=True)
def row_iq1_m(W: tl.View[ggml.IQ1_M, (K,)], X: tl.View[ggml.Q8_K, (K,)],
              grid: tl.View[ggml.I8X8, (2048, 8)], DEFERRED: tl.constexpr):
    return compute_iq1_m(W, X, grid, K, DEFERRED)


@triton.jit(noinline=True)
def row_iq1_s(W: tl.View[ggml.IQ1_S, (K,)], X: tl.View[ggml.Q8_K, (K,)],
              grid: tl.View[ggml.I8X8, (2048, 8)]):
    return compute_iq1_s(W, X, grid, K)


@triton.jit(noinline=True)
def row_iq2_xs(W: tl.View[ggml.IQ2_XS, (K,)], X: tl.View[ggml.Q8_K, (K,)],
               grid: tl.View[ggml.I8X8, (512, 8)], signs: tl.View[ggml.I8X8, (128, 8)]):
    return compute_iq2_xs(W, X, grid, signs, K)


@triton.jit(noinline=True)
def row_iq2_xxs(W: tl.View[ggml.IQ2_XXS, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                grid: tl.View[ggml.I8X8, (256, 8)], signs: tl.View[ggml.I8X8, (128, 8)]):
    return compute_iq2_xxs(W, X, grid, signs, K)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq1_m(W: tl.View[ggml.IQ1_M, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X8, (2048, 8)],
           Y: tl.View[tl.f32, (M, N)], *, DEFERRED: tl.constexpr = False):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq1_m(W[column], Xq[row], grid, DEFERRED))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq1_s(W: tl.View[ggml.IQ1_S, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X8, (2048, 8)],
           Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq1_s(W[column], Xq[row], grid))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xs(W: tl.View[ggml.IQ2_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X8, (512, 8)],
            signs: tl.View[ggml.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq2_xs(W[column], Xq[row], grid, signs))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xxs(W: tl.View[ggml.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
             Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X8, (256, 8)],
             signs: tl.View[ggml.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq2_xxs(W[column], Xq[row], grid, signs))


@triton.jit(noinline=True)
def row_iq2_xxs_tensor(W: tl.View[ggml.IQ2_XXS, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                      grid: tl.View[ggml.I8X8, (256, 8)], signs: tl.View[ggml.I8X8, (128, 8)]):
    return compute_iq2_xxs_tensor(W, X, grid, signs)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xxs_tensor(W: tl.View[ggml.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
                   Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X8, (256, 8)],
                   signs: tl.View[ggml.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq2_xxs_tensor(W[column], Xq[row], grid, signs))


@triton.jit(noinline=True)
def row_iq3_xxs_tensor(W: tl.View[ggml.IQ3_XXS, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                      grid: tl.View[ggml.I8X4, (256, 4)], signs: tl.View[ggml.I8X8, (128, 8)]):
    return compute_iq3_xxs_tensor(W, X, grid, signs)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq3_xxs_tensor(W: tl.View[ggml.IQ3_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
                   Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X4, (256, 4)],
                   signs: tl.View[ggml.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_iq3_xxs_tensor(W[column], Xq[row], grid, signs))
