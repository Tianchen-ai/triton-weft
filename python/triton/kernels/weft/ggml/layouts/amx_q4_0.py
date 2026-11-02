"""Author-declared storage layouts used by the amx q4 0 kernels."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class AMXQ4_0:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    q: tl.u4[32] @ tl.grouped(elements=8) @ tl.bit_layers(elements=4, order=tl.lo_first)
    d: tl.f16


@triton.derive
def AMXQ4_0Rows(W: tl.View[AMXQ4_0, (N, K)], *, rows: tl.static[int]) -> tl.View[AMXQ4_0Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"q": 4, "d": 2})


@triton.encoding
class BlockScale32:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 4
    elements = 32
    value: tl.f32


@triton.derive
def BlockScale32Rows(S: tl.View[BlockScale32, (M, K)], *, rows: tl.static[int]) -> tl.View[BlockScale32Rows[rows], (M, K)]:
    return tl.interleave(S, rows=rows, field_chunks={"value": 4})
