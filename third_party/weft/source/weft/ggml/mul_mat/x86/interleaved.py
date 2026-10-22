from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as tl
from weft.ggml.mul_mat.x86.blocks32 import project
from weft.ggml.quantize.x86.blocks32 import BlockSum32


@triton.derive
def Q4_0Rows(W: tl.View[ggml.Q4_0, (N, K)], *, rows: tl.static[int]) -> tl.View[Q4_0Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})


@triton.derive
def Q4_1Rows(W: tl.View[ggml.Q4_1, (N, K)], *, rows: tl.static[int]) -> tl.View[Q4_1Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "m": 2, "q": 4})


@triton.derive
def Q5_0Rows(W: tl.View[ggml.Q5_0, (N, K)], *, rows: tl.static[int]) -> tl.View[Q5_0Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "qh": 4, "q": 4})


@triton.derive
def Q5_1Rows(W: tl.View[ggml.Q5_1, (N, K)], *, rows: tl.static[int]) -> tl.View[Q5_1Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "m": 2, "qh": 4, "q": 4})


@triton.derive
def Q8_0Rows(W: tl.View[ggml.Q8_0, (N, K)], *, rows: tl.static[int]) -> tl.View[Q8_0Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q8_0(W: tl.View[Q8_0Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
         Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 0, False, False, BM, 32, True, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q8_0_prefill(W: tl.View[Q8_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
                 Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 0, False, False, BM, 16, True, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q5_0(W: tl.View[Q5_0Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
         Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 16, True, False, BM, 32, False, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_1(W: tl.View[Q5_1Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 0, True, True, BM, 32)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q5_0_prefill(W: tl.View[Q5_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
                 Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 16, True, False, BM, 16, False, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_1_prefill(W: tl.View[Q5_1Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 0, True, True, BM, 16)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q4_0(W: tl.View[Q4_0Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
         Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 8, False, False, BM, 32, False, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_1(W: tl.View[Q4_1Rows[32], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 1):
    project(W, X, Xq, Y, M, N, K, 0, False, True, BM, 32)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Sums": "sums", "Y": "output"})
def q4_0_prefill(W: tl.View[Q4_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml.Q8_0, (M, K)], Sums: tl.View[BlockSum32, (M, K)],
                 Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 8, False, False, BM, 16, False, Sums)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_1_prefill(W: tl.View[Q4_1Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4):
    project(W, X, Xq, Y, M, N, K, 0, False, True, BM, 16)
