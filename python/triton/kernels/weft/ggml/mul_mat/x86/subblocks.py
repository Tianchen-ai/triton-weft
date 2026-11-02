from __future__ import annotations

from triton.integrations import ggml_formats
from ...blocks.quantize import q8_0 as quant_q8_0
from ...blocks.quantize import q8_k as quant_q8_k
import triton
import triton.language as tl
from ...vec_dot.x86.subblocks import compute_q1_0, compute_nvfp4
from .codebook import project as project_table
from ...vec_dot.x86.ternary import compute as compute_ternary


@triton.jit
def project(W, X, Xq, Y, M, N, QUANTIZE: tl.constexpr, DOT: tl.constexpr):
    QUANTIZE(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], DOT(W[column], Xq[row]))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q1_0(W: tl.View[ggml_formats.Q1_0, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project(W, X, Xq, Y, M, N, quant_q8_0.quantize_matrix, compute_q1_0)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q1_0_specialized(W: tl.View[ggml_formats.Q1_0, (N, K)], X: tl.View[tl.f32, (M, K)],
                     Xq: tl.View[ggml_formats.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    project(W, X, Xq, Y, M, N, quant_q8_0.quantize_matrix, compute_q1_0)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def nvfp4(W: tl.View[ggml_formats.NVFP4, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
           scale: tl.View[tl.f32, (256,)], Y: tl.View[tl.f32, (M, N)]):
    project_table(W, X, Xq, codebook, scale, Y, M, N,
                  quant_q8_0.quantize_matrix, compute_nvfp4, True)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def tq1_0(W: tl.View[ggml_formats.TQ1_0, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_K, (M, K)], powers: tl.View[tl.u32, (5,)],
          Y: tl.View[tl.f32, (M, N)], *, RECORDS: tl.constexpr = 2):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], compute_ternary(W[column], Xq[row], powers, True, RECORDS))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def nvfp4_specialized(W: tl.View[ggml_formats.NVFP4, (N, K)], X: tl.View[tl.f32, (M, K)],
                      Xq: tl.View[ggml_formats.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
                      scale: tl.View[tl.f32, (256,)], Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    project_table(W, X, Xq, codebook, scale, Y, M, N,
                  quant_q8_0.quantize_matrix, compute_nvfp4, True)
