"""Author-declared storage layouts used by the ime2 kernels."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class IME2Weight8:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    q: tl.i8[32]


@triton.encoding
class IME2Activation8:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f32
    neg_sum: tl.i16
    q: tl.i8[32]


@triton.encoding
class IME2Weight4Affine:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 32
    d: tl.f16
    zp: tl.u8
    q: tl.u4[32]


@triton.derive
def IME2Weight4AffineRows(W: tl.View[IME2Weight4Affine, (N, K)], *, rows: tl.static[int]) -> tl.View[IME2Weight4AffineRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "zp": 1, "q": 16})


@triton.derive
def IME2Weight8Rows(W: tl.View[IME2Weight8, (N, K)], *, rows: tl.static[int]) -> tl.View[IME2Weight8Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 32})


@triton.derive
def IME2Activation8Rows(X: tl.View[IME2Activation8, (M, K)], *, rows: tl.static[int]) -> tl.View[IME2Activation8Rows[rows], (M, K)]:
    return tl.interleave(X, rows=rows, field_chunks={"d": 4, "neg_sum": 2, "q": 32})
