"""Author-declared storage layouts used by the amx k kernels."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class AMXQ4_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    q: tl.u4[256] @ tl.grouped(elements=8) @ tl.bit_layers(elements=4, order=tl.lo_first)
    scales: tl.u8[8]
    mins: tl.u8[8]
    d: tl.f16
    dmin: tl.f16


@triton.derive
def AMXQ4_KRows(W: tl.View[AMXQ4_K, (N, K)], *, rows: tl.static[int]) -> tl.View[AMXQ4_KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"q": 4, "scales": 1, "mins": 2, "d": 2, "dmin": 2})


@triton.encoding
class AMXQ6_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    low: tl.u4[256] @ tl.grouped(elements=8) @ tl.bit_layers(elements=4, order=tl.lo_first)
    high: tl.u2[256] @ tl.grouped(elements=16) @ tl.bit_layers(elements=4, order=tl.lo_first)
    scales: tl.i8[16]
    d: tl.f16


@triton.derive
def AMXQ6_KRows(W: tl.View[AMXQ6_K, (N, K)], *, rows: tl.static[int]) -> tl.View[AMXQ6_KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"low": 4, "high": 4, "scales": 1, "d": 2})
