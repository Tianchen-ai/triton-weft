from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_dequantize_q6_k(W: tl.View[ggml_formats.Q6_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        with tl.level.subtiles(kb, extent=16) as sub:
            q = tl.i32(w.ql[sub]) | tl.i32(w.qh[sub]) << tl.u32(4)
            q -= tl.i32(32)
            tl.store(Y[kb][sub], tl.f32(w.d) * tl.f32(w.scales[sub]) * tl.f32(q))
