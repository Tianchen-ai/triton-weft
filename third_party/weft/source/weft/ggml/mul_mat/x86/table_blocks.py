from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as tl
from weft.ggml.mul_mat.x86.blocks32 import project
from weft.ggml.quantize.x86.blocks32 import BlockSum32


@triton.derive
def IQ4Rows(W: tl.View[ggml.IQ4_NL, (N, K)], *, rows: tl.static[int]) -> tl.View[IQ4Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})


@triton.derive
def MXRows(W: tl.View[ggml.MXFP4, (N, K)], *, rows: tl.static[int]) -> tl.View[MXRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"e": 1, "q": 4})


@triton.jit
def decode(q, table):
    return tl.lookup(table, tl.u8(q), bounds="in_bounds")


@triton.jit
def exponent(w, scale):
    return tl.lookup(scale, tl.u32(tl.load(w.e)), bounds="in_bounds")


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Sums": "sums", "Y": "output"})
def iq4_nl(W: tl.View[IQ4Rows[BN], (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
           Sums: tl.View[BlockSum32, (M, K)], Y: tl.View[tl.f32, (M, N)],
           BM: tl.constexpr = 4, BN: tl.constexpr = 16):
    table = tl.load(codebook)
    project(W, X, Xq, Y, M, N, K, 0, False, False, BM, BN, True, Sums, decode, None, table)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Sums": "sums", "Y": "output"})
def mxfp4(W: tl.View[MXRows[BN], (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
          scale: tl.View[tl.f32, (256,)], Sums: tl.View[BlockSum32, (M, K)],
          Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4, BN: tl.constexpr = 16):
    table = tl.load(codebook)
    project(W, X, Xq, Y, M, N, K, 0, False, False, BM, BN, True, Sums, decode, exponent, table, scale)
