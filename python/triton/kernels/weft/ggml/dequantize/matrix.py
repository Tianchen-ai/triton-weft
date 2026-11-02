from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from .tiles import decode_codebook_records, decode_records
from . import tiles
from .radix import ternary_record


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq4_nl(W: tl.View[ggml_formats.IQ4_NL, (rows, K)], codebook: tl.View[tl.i8, (16,)],
           Y: tl.View[tl.f32, (rows, K)], RECORDS: tl.constexpr = 2, CACHE: tl.constexpr = ""):
    for row in range(rows):
        decode_codebook_records(W[row], codebook, None, Y[row], K, RECORDS, False, CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def mxfp4(W: tl.View[ggml_formats.MXFP4, (rows, K)], codebook: tl.View[tl.i8, (16,)],
          scale: tl.View[tl.f32, (256,)], Y: tl.View[tl.f32, (rows, K)], RECORDS: tl.constexpr = 2, CACHE: tl.constexpr = ""):
    for row in range(rows):
        decode_codebook_records(W[row], codebook, scale, Y[row], K, RECORDS, True, CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_0(W: tl.View[ggml_formats.Q4_0, (rows, K)], Y: tl.View[tl.f32, (rows, K)], RECORDS: tl.constexpr = 1,
         CACHE: tl.constexpr = ""):
    for row in range(rows):
        decode_records(W[row], Y[row], K, RECORDS, 8, False, False, CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (rows, K)], grid: tl.View[tl.i8, (16384,)],
          Y: tl.View[tl.f32, (rows, K)], ENTRIES: tl.constexpr = 8, CACHE: tl.constexpr = ""):
    for row in range(rows):
        tiles.iq1_s(W[row], grid, Y[row], ENTRIES=ENTRIES, CACHE=CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q2_k(W: tl.View[ggml_formats.Q2_K, (rows, K)], Y: tl.View[tl.f32, (rows, K)],
         GROUPS: tl.constexpr = 2, CACHE: tl.constexpr = ""):
    for row in range(rows):
        tiles.q2_k(W[row], Y[row], GROUPS=GROUPS, CACHE=CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q8_0(W: tl.View[ggml_formats.Q8_0, (rows, K)], Y: tl.View[tl.f32, (rows, K)],
         LANES: tl.constexpr = 16, CACHE: tl.constexpr = ""):
    for row in range(rows):
        tiles.q8_0(W[row], Y[row], LANES=LANES, CACHE=CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def tq1_0(W: tl.View[ggml_formats.TQ1_0, (rows, K)], Y: tl.View[tl.f32, (rows, K)], CACHE: tl.constexpr = ""):
    for row in range(rows):
        with tl.level.blocks(K, extent=256) as block:
            w = tl.load(W[row, block])
            value = tl.f32(tl.i8(ternary_record(w)) - tl.i8(1)) * tl.f32(w.d)
            tl.store(Y[row, block], value, cache_modifier=CACHE)
