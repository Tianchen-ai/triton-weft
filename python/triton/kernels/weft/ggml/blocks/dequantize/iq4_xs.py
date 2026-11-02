from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def _signed_scale(raw, zero: int = 0):
    return tl.i32(raw) - tl.i32(zero)


@triton.jit(model="weft")
def row_dequantize_iq4_xs(
    W: tl.View[ggml_formats.IQ4_XS, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    Y: tl.View[tl.f32, (K,)],
):
    table = tl.stage(tl.load(codebook))
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        sub_index = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as sub:
            low = qf.extract_bits(
                w.scales_l[sub_index // tl.index(2)],
                sub_index % tl.index(2) * tl.index(4),
                4,
            )
            high = qf.extract_bits(w.scales_h, sub_index * tl.index(2), 2)
            scale = _signed_scale(low | high << tl.u32(4), zero=32)
            q = qf.small_nonlinear_lookup(table, w.q[sub])
            tl.store(Y[kb][sub], qf.iq_codebook(tl.i32(q), tl.f32(w.d) * tl.f32(scale)))
            sub_index += tl.index(1)
