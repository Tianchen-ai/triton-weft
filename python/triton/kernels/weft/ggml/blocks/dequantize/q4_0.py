from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_dequantize_q4_0(W: tl.View[ggml_formats.Q4_0, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=32) as kb:
        w = tl.load(W[kb])
        tl.store(Y[kb], qf.symmetric_integer(w.q, w.d, zero=8))
