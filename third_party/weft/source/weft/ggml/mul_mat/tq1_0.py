from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
from kernels.vec_dot.tq1_0_q8_k import compute
from weft.ggml.vec_dot.ternary import compute_tq1
import triton
import triton.language as tl


@triton.jit(noinline=True)
def row_dot(W: tl.View[ggml.TQ1_0, (K,)], X: tl.View[ggml.Q8_K, (K,)],
            powers: tl.View[tl.u32, (5,)]):
    return compute(W, X, powers)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection(W: tl.View[ggml.TQ1_0, (N, K)], X: tl.View[tl.f32, (M, K)],
               Xq: tl.View[ggml.Q8_K, (M, K)], powers: tl.View[tl.u32, (5,)],
               Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_dot(W[column], Xq[row], powers))


@triton.jit(noinline=True)
def bounded_row_dot(W: tl.View[ggml.TQ1_0, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                    powers: tl.View[tl.u32, (5,)]):
    return compute_tq1(W, X, powers)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection_bounded(W: tl.View[ggml.TQ1_0, (N, K)], X: tl.View[tl.f32, (M, K)],
                       Xq: tl.View[ggml.Q8_K, (M, K)], powers: tl.View[tl.u32, (5,)],
                       Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], bounded_row_dot(W[column], Xq[row], powers))
