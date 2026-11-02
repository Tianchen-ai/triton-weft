from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_dequantize_q1_0(W: tl.View[ggml_formats.Q1_0, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=128) as kb:
        w = tl.load(W[kb])
        q = tl.i32(tl.i8(w.q) * tl.i8(2) - tl.i8(1))
        tl.store(Y[kb], qf.ternary_radix(q, w.d))
