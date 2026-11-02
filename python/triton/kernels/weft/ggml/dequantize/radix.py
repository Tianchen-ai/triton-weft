from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def ternary_record(w):
    """Reconstruct a complete radix-3 record with its declared byte wrap."""
    lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
    offset = tl.where(lane < tl.u32(160), lane % tl.u32(32),
                      tl.u32(32) + (lane - tl.u32(160)) % tl.u32(16))
    packed = tl.u8(w.q[offset])
    packed = tl.where(lane < tl.u32(240), packed, tl.u8(w.qh[lane % tl.u32(4)]))
    digit = tl.where(lane < tl.u32(160), lane // tl.u32(32),
                     tl.where(lane < tl.u32(240), (lane - tl.u32(160)) // tl.u32(16),
                              (lane - tl.u32(240)) // tl.u32(4)))
    power = tl.where(digit == tl.u32(0), tl.u8(1),
                     tl.where(digit == tl.u32(1), tl.u8(3),
                              tl.where(digit == tl.u32(2), tl.u8(9),
                                       tl.where(digit == tl.u32(3), tl.u8(27), tl.u8(81)))))
    wrapped = power * packed
    return tl.u2((tl.u16(wrapped) * tl.u16(3)) >> tl.u16(8))
