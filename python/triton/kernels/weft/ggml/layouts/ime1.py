"""Author-declared storage layouts used by the ime1 kernels."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class IMEQ4Centered:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    q: tl.u4[32] @ tl.grouped(elements=16) @ tl.bit_layers(elements=8, order=tl.lo_first)


@triton.encoding
class IMEQ4Affine:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    elements = 32
    d: tl.f16
    zp: tl.u8
    q: tl.u4[32] @ tl.grouped(elements=16) @ tl.bit_layers(elements=8, order=tl.lo_first)


@triton.encoding
class IMEQ8:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 4
    elements = 32
    d: tl.f32
    q: tl.i8[32]


@triton.derive
def IMEQ4Centered_I(W: tl.View[IMEQ4Centered, (N, K)], *, rows: tl.static[int]) -> tl.View[IMEQ4Centered_I[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 8})


@triton.derive
def IMEQ4Affine_I(W: tl.View[IMEQ4Affine, (N, K)], *, rows: tl.static[int]) -> tl.View[IMEQ4Affine_I[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "zp": 1, "q": 8})


@triton.derive
def IMEQ8_I(X: tl.View[IMEQ8, (M, K)], *, rows: tl.static[int]) -> tl.View[IMEQ8_I[rows], (M, K)]:
    return tl.interleave(X, rows=rows, field_chunks={"d": 4, "q": 8})
