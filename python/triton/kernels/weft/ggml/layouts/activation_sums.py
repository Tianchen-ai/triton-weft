"""Author-declared storage layouts used by the activation sums kernels."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class BlockSum32:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    value: tl.i16
