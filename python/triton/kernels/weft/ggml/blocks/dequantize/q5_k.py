from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_q5_k(W: tl.View[ggml_formats.Q5_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        with tl.level.subtiles(kb, extent=32) as sub:
            q = tl.i32(w.q[sub]) | tl.i32(w.qh[sub]) << tl.u32(4)
            tl.store(Y[kb][sub], qf.k_superblock(q, w.sc[sub], w.d, w.m[sub], w.dmin))
