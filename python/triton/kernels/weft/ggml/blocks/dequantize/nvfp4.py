from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_dequantize_nvfp4(
    W: tl.View[ggml_formats.NVFP4, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    scale: tl.View[tl.f32, (256,)],
    Y: tl.View[tl.f32, (K,)],
):
    table = tl.stage(tl.load(codebook))
    with tl.level.blocks(K, extent=64) as kb:
        w = tl.load(W[kb])
        sub_index = tl.index(0)
        with tl.level.subtiles(kb, extent=16) as sub:
            decoded_scale = qf.exponent_scale(scale, w.d[sub_index])
            q = qf.small_nonlinear_lookup(table, w.q[sub])
            tl.store(Y[kb][sub], qf.fp4_codebook(q, decoded_scale))
            sub_index += tl.index(1)
