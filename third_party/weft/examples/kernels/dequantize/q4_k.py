from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import triton
import triton.language as wl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_q4_k(W: wl.View[ggml.Q4_K, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        with wl.level.subtiles(kb, extent=32) as sub:
            wl.store(Y[kb][sub], qf.k_superblock(w.q[sub], w.sc[sub], w.d, w.m[sub], w.dmin))
