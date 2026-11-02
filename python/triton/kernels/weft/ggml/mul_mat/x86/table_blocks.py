from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from triton.kernels.weft.ggml.layouts.lookup_rows import IQ4Rows, MXRows
from .blocks32 import project
from triton.kernels.weft.ggml.layouts.activation_sums import BlockSum32


@triton.jit
def decode(q, table):
    return tl.lookup(table, tl.u8(q), bounds="in_bounds")


@triton.jit
def exponent(w, scale):
    return tl.lookup(scale, tl.u32(tl.load(w.e)), bounds="in_bounds")


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Sums": "sums", "Y": "output"})
def iq4_nl(W: tl.View[IQ4Rows[BN], (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
           Sums: tl.View[BlockSum32, (M, K)], Y: tl.View[tl.f32, (M, N)],
           BM: tl.constexpr = 4, BN: tl.constexpr = 16):
    table = tl.load(codebook)
    project(W, X, Xq, Y, M, N, K, 0, False, False, BM, BN, True, Sums, decode, None, table)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Sums": "sums", "Y": "output"})
def mxfp4(W: tl.View[MXRows[BN], (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
          scale: tl.View[tl.f32, (256,)], Sums: tl.View[BlockSum32, (M, K)],
          Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4, BN: tl.constexpr = 16):
    table = tl.load(codebook)
    project(W, X, Xq, Y, M, N, K, 0, False, False, BM, BN, True, Sums, decode, exponent, table, scale)
