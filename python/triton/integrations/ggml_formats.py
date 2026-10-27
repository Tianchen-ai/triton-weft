"""GGML base storage encodings and the shared Q4_K row derivation.

These declarations describe bytes and logical fields, not reference kernels.
GGML-specific author layouts for benchmark candidates live in Bench's
weft_bench.kernels.ggml.layouts package. Applications prepare persistent storage
explicitly before passing it to a kernel or binding a native GGML call site.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class Q1_0:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 128
    d: tl.f16
    q: tl.u1[128] @ tl.grouped(elements=8) @ tl.bit_layers(elements=1, order=tl.lo_first)


@triton.encoding
class Q4_0:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.encoding
class Q4_1:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    m: tl.f16
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.encoding
class Q5_0:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    qh: tl.u1[32] @ tl.grouped(elements=8) @ tl.bit_layers(elements=1, order=tl.lo_first)
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.encoding
class Q5_1:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    m: tl.f16
    qh: tl.u1[32] @ tl.grouped(elements=8) @ tl.bit_layers(elements=1, order=tl.lo_first)
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.encoding
class Q8_0:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    q: tl.i8[32]


@triton.encoding
class Q2_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    scales: tl.u8[16]
    q: tl.u2[256] @ tl.grouped(elements=128) @ tl.bit_layers(elements=32, order=tl.lo_first)
    d: tl.f16
    dmin: tl.f16


@triton.encoding
class Q3_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    hmask: tl.u1[256] @ tl.grouped(elements=256) @ tl.bit_layers(elements=32, order=tl.lo_first)
    q: tl.u2[256] @ tl.grouped(elements=128) @ tl.bit_layers(elements=32, order=tl.lo_first)
    scale_low: tl.u4[16] @ tl.grouped(elements=16) @ tl.bit_layers(elements=8, order=tl.lo_first)
    scale_high: tl.u2[16] @ tl.grouped(elements=16) @ tl.bit_layers(elements=4, order=tl.lo_first)
    d: tl.f16


@triton.encoding
class Q4_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    dmin: tl.f16
    sc: tl.u6[8] @ tl.pack_fields(group=4, fields=2, low_bits=4, order=tl.lo_first)
    m: tl.u6[8] @ tl.pack_fields(group=4, fields=2, low_bits=4, order=tl.lo_first)
    q: tl.u4[256] @ tl.grouped(elements=64) @ tl.bit_layers(elements=32, order=tl.lo_first)


@triton.encoding
class Q5_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    dmin: tl.f16
    sc: tl.u6[8] @ tl.pack_fields(group=4, fields=2, low_bits=4, order=tl.lo_first)
    m: tl.u6[8] @ tl.pack_fields(group=4, fields=2, low_bits=4, order=tl.lo_first)
    qh: tl.u1[256] @ tl.grouped(elements=256) @ tl.bit_layers(elements=32, order=tl.lo_first)
    q: tl.u4[256] @ tl.grouped(elements=64) @ tl.bit_layers(elements=32, order=tl.lo_first)


@triton.encoding
class Q6_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    elements = 256
    ql: tl.u4[256] @ tl.grouped(elements=128) @ tl.bit_layers(elements=64, order=tl.lo_first)
    qh: tl.u2[256] @ tl.grouped(elements=128) @ tl.bit_layers(elements=32, order=tl.lo_first)
    scales: tl.i8[16] @ tl.grouped(elements=16) @ tl.bit_layers(elements=16, order=tl.lo_first)
    d: tl.f16


@triton.encoding
class IQ1_S:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    q: tl.u8[32]
    qh: tl.u16[8]


@triton.encoding
class IQ1_M:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    elements = 256
    q: tl.u8[32]
    qh: tl.u8[16]
    scales: tl.u8[8]


@triton.encoding
class IQ2_S:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    q: tl.u8[32]
    signs: tl.u1[256] @ tl.grouped(elements=8) @ tl.bit_layers(elements=1, order=tl.lo_first)
    qh: tl.u8[8]
    scales: tl.u8[8]


@triton.encoding
class IQ2_XS:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    q: tl.u16[32]
    scales: tl.u8[8]


@triton.encoding
class IQ2_XXS:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    q: tl.u16[32]


@triton.encoding
class I8X8:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 8
    elements = 8
    values: tl.i8[8]


@triton.encoding
class I8X4:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 4
    elements = 4
    values: tl.i8[4]


@triton.encoding
class IQ3_S:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    q: tl.u8[64]
    qh: tl.u1[64] @ tl.grouped(elements=8) @ tl.bit_layers(elements=1, order=tl.lo_first)
    signs: tl.u1[256] @ tl.grouped(elements=8) @ tl.bit_layers(elements=1, order=tl.lo_first)
    scales: tl.u4[8] @ tl.grouped(elements=2) @ tl.bit_layers(elements=1, order=tl.lo_first)


@triton.encoding
class IQ3_XXS:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    q: tl.u8[64]
    metadata: tl.u32[8]


@triton.encoding
class IQ4_NL:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.encoding
class IQ4_XS:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    d: tl.f16
    scales_h: tl.u16
    scales_l: tl.u8[4]
    q: tl.u4[256] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.encoding
class TQ1_0:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    q: tl.u8[48]
    qh: tl.u8[4]
    d: tl.f16


@triton.encoding
class TQ2_0:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    q: tl.u2[256] @ tl.grouped(elements=128) @ tl.bit_layers(elements=32, order=tl.lo_first)
    d: tl.f16


@triton.encoding
class MXFP4:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    elements = 32
    e: tl.u8
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.encoding
class NVFP4:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    elements = 64
    d: tl.u8[4]
    q: tl.u4[64] @ tl.grouped(elements=16) @ tl.bit_layers(elements=8, order=tl.lo_first)


@triton.encoding
class Q8_1:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    s: tl.f16
    q: tl.i8[32]


@triton.encoding
class Q8_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 4
    elements = 256
    ds: tl.f32
    q: tl.i8[256]
    bsum: tl.i16[16]


@triton.derive
def Q4K_I(
    W: tl.View[Q4_K, (M, K)], *, rows: tl.static[int]
) -> tl.View[Q4K_I[rows], (M, K)]:
    return tl.interleave(W, rows=rows)
