from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_q3_k(W: tl.View[ggml_formats.Q3_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        with tl.level.subtiles(kb, extent=16) as sub:
            scale = (
                tl.i32(w.scale_low[sub]) | tl.i32(w.scale_high[sub]) << tl.u32(4)
            ) - tl.i32(32)
            q = tl.i32(w.q[sub]) + tl.i32(w.hmask[sub]) * tl.i32(4) - tl.i32(4)
            tl.store(Y[kb][sub], tl.f32(w.d) * tl.f32(scale) * tl.f32(q))
