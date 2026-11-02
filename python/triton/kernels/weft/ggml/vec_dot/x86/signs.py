from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def even_parity_signs(index, payload):
    """Expand seven stored sign bits and their even-parity eighth bit."""
    nibble = tl.arange(0, 16, dtype=tl.u8, axis="nibble")
    table = ((nibble ^ (nibble >> 1) ^ (nibble >> 2) ^ (nibble >> 3)) & 1) << 7
    folded = (tl.u8(index) ^ (tl.u8(index) >> 4)) & tl.u8(15)
    bits = tl.u8(index) | tl.lookup(table, folded, bounds="in_bounds")
    negative = (bits & (tl.u8(1) << tl.u8(payload))) != 0
    return tl.where(negative, tl.i8(-1), tl.i8(1))
