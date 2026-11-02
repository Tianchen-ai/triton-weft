from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from triton.kernels.weft.ggml.layouts.interleaved_rows import Q4_0Rows, Q4_1Rows, Q5_0Rows, Q5_1Rows, Q8_0Rows
from .blocks32 import project
from triton.kernels.weft.ggml.layouts.activation_sums import BlockSum32


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q8_0(W: tl.View[Q8_0Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
         Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 0, False, False, BM, 32, True, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q8_0_prefill(W: tl.View[Q8_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml_formats.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
                 Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 0, False, False, BM, 16, True, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q5_0(W: tl.View[Q5_0Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
         Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 16, True, False, BM, 32, False, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_1(W: tl.View[Q5_1Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 0, True, True, BM, 32)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q5_0_prefill(W: tl.View[Q5_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml_formats.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
                 Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 16, True, False, BM, 16, False, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_1_prefill(W: tl.View[Q5_1Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml_formats.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 0, True, True, BM, 16)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q4_0(W: tl.View[Q4_0Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
         Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1, BN: tl.constexpr = 32):
    project(W, X, Xq, Y, M, N, K, 8, False, False, BM, BN, False, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_1(W: tl.View[Q4_1Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 0, False, True, BM, 32)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q4_0_prefill(W: tl.View[Q4_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml_formats.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
                 Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 8, False, False, BM, 16, False, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_1_prefill(W: tl.View[Q4_1Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml_formats.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 0, False, True, BM, 16)
