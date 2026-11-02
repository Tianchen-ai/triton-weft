from __future__ import annotations

from triton.integrations import ggml_formats
from ...blocks.quantize import q8_k as quant_q8_k
import triton
import triton.language as tl
from ...vec_dot.x86.q4_k_q8_k import compute as dot_q4_k
from ...vec_dot.x86.q6_k_q8_k import compute as dot_q6_k
from ...vec_dot.x86.grouped_k import compute_q2, compute_q3, compute_q5
from ...vec_dot.x86.scaled_records import compute as dot_scaled_record
from ...vec_dot.x86.scaled_records import compute_iq1_s
from ...vec_dot.x86.scaled_records import compute_grouped as dot_grouped_record
from .scaled_records import decode_iq1_m
from ...vec_dot.x86.ternary import compute as dot_ternary


@triton.jit
def project(W, X, Xq, Y, M, N, DOT: tl.constexpr, LANES: tl.constexpr, ROWS: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        if ROWS:
            with tl.level.cols(N, group=ROWS) as column:
                tl.store(Y[row, column], DOT(W, Xq[row], LANES, ROWS, column))
        else:
            for column in range(N):
                tl.store(Y[row, column], DOT(W[column], Xq[row], LANES))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_k(W: tl.View[ggml_formats.Q4_K, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
         *, LANES: tl.constexpr, ROWS: tl.constexpr = 0):
    project(W, X, Xq, Y, M, N, dot_q4_k, LANES, ROWS)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q6_k(W: tl.View[ggml_formats.Q6_K, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
         *, LANES: tl.constexpr, ROWS: tl.constexpr = 0):
    project(W, X, Xq, Y, M, N, dot_q6_k, LANES, ROWS)


@triton.jit
def project_grouped(W, X, Xq, Y, M, N, DOT: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], DOT(W[column], Xq[row]))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q2_k(W: tl.View[ggml_formats.Q2_K, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project_grouped(W, X, Xq, Y, M, N, compute_q2)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q3_k(W: tl.View[ggml_formats.Q3_K, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project_grouped(W, X, Xq, Y, M, N, compute_q3)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_k(W: tl.View[ggml_formats.Q5_K, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project_grouped(W, X, Xq, Y, M, N, compute_q5)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q2_k_specialized(W: tl.View[ggml_formats.Q2_K, (N, K)], X: tl.View[tl.f32, (M, K)],
                     Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    project_grouped(W, X, Xq, Y, M, N, compute_q2)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_k_specialized(W: tl.View[ggml_formats.Q5_K, (N, K)], X: tl.View[tl.f32, (M, K)],
                     Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    project_grouped(W, X, Xq, Y, M, N, compute_q5)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq1_m(W: tl.View[ggml_formats.IQ1_M, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
          f16_bits: tl.View[tl.f32, (65536,)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], dot_scaled_record(W[column], Xq[row], grid, f16_bits, decode_iq1_m))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def tq2_0(W: tl.View[ggml_formats.TQ2_0, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], dot_ternary(W[column], Xq[row], None, False))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
          Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], compute_iq1_s(W[column], Xq[row], grid))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xxs(W: tl.View[ggml_formats.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (256, 8)],
           Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], dot_grouped_record(W[column], Xq[row], grid, False))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_xs(W: tl.View[ggml_formats.IQ2_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (512, 8)],
           Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], dot_grouped_record(W[column], Xq[row], grid, True))
