from __future__ import annotations

from triton.integrations import ggml_formats
from ..blocks.quantize import q8_k as quant_q8_k
from ..blocks.vec_dot.iq1_s_q8_k import quantized_vec_dot_iq1_s_q8_k as iq1_s_dot
from ..blocks.vec_dot.iq2_xs_q8_k import quantized_vec_dot_iq2_xs_q8_k as iq2_xs_dot
import triton
import triton.language as tl


@triton.jit(noinline=True)
def iq1_s_row(W: tl.View[ggml_formats.IQ1_S, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
              grid: tl.View[ggml_formats.I8X8, (2048, 8)], Y: tl.View[tl.f32, (1,)]):
    iq1_s_dot(W, X, grid, Y)


@triton.jit(noinline=True)
def iq2_xs_row(W: tl.View[ggml_formats.IQ2_XS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
               grid: tl.View[ggml_formats.I8X8, (512, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)],
               Y: tl.View[tl.f32, (1,)]):
    iq2_xs_dot(W, X, grid, signs, Y)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
          Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            iq1_s_row(W[column], Xq[row], grid,
                      tl.subview(Y[row], offsets=(column,), extents=(1,)))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xs(W: tl.View[ggml_formats.IQ2_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (512, 8)],
           signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            iq2_xs_row(W[column], Xq[row], grid, signs,
                       tl.subview(Y[row], offsets=(column,), extents=(1,)))
