from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_dequantize_mxfp4(
    W: tl.View[ggml_formats.MXFP4, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    scale: tl.View[tl.f32, (256,)],
    Y: tl.View[tl.f32, (K,)],
):
    table = tl.stage(tl.load(codebook))
    with tl.level.blocks(K, extent=32) as kb:
        w = tl.load(W[kb])
        decoded_scale = qf.exponent_scale(scale, w.e)
        q = qf.small_nonlinear_lookup(table, w.q)
        tl.store(Y[kb], qf.fp4_codebook(q, decoded_scale))
