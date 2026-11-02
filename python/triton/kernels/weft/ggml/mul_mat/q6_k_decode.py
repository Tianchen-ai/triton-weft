from __future__ import annotations

from triton.integrations import ggml_formats
from ..blocks.quantize import q8_k as quant_q8_k
import triton
import triton.language as tl

from ..vec_dot.q6_k import block_scaled, stream_scale_groups
from ..blocks.vec_dot.q6_k_q8_k import compute as compute_scale_groups


@triton.jit(noinline=True)
def row_dot(W: tl.View[ggml_formats.Q6_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
            LANES: tl.constexpr, PLANES: tl.constexpr):
    return block_scaled(W, X, LANES, PLANES)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection(W: tl.View[ggml_formats.Q6_K, (N, K)], X: tl.View[tl.f32, (1, K)],
               Xq: tl.View[ggml_formats.Q8_K, (1, K)], Y: tl.View[tl.f32, (1, N)],
               *, LANES: tl.constexpr = 64, PLANES: tl.constexpr = 2):
    quant_q8_k.quantize_row(X[0], Xq[0])
    for column in range(N):
        tl.store(Y[0, column], row_dot(W[column], Xq[0], LANES, PLANES))


@triton.jit(noinline=True)
def scale_group_row(W: tl.View[ggml_formats.Q6_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)]):
    return compute_scale_groups(W, X)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection_scale_groups(W: tl.View[ggml_formats.Q6_K, (N, K)], X: tl.View[tl.f32, (1, K)],
                            Xq: tl.View[ggml_formats.Q8_K, (1, K)], Y: tl.View[tl.f32, (1, N)]):
    quant_q8_k.quantize_row(X[0], Xq[0])
    for column in range(N):
        tl.store(Y[0, column], scale_group_row(W[column], Xq[0]))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection_rows(W: tl.View[ggml_formats.Q6_K, (N, K)], X: tl.View[tl.f32, (M, K)],
                    Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], scale_group_row(W[column], Xq[row]))


@triton.jit(noinline=True)
def streamed_row(W: tl.View[ggml_formats.Q6_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                 GROUPS: tl.constexpr):
    return stream_scale_groups(W, X, GROUPS)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection_streamed(W: tl.View[ggml_formats.Q6_K, (N, K)], X: tl.View[tl.f32, (1, K)],
                        Xq: tl.View[ggml_formats.Q8_K, (1, K)], Y: tl.View[tl.f32, (1, N)],
                        *, GROUPS: tl.constexpr = 1):
    quant_q8_k.quantize_row(X[0], Xq[0])
    for column in range(N):
        tl.store(Y[0, column], streamed_row(W[column], Xq[0], GROUPS))
