from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_q4_k(W: tl.View[ggml_formats.Q4_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        with tl.level.subtiles(kb, extent=32) as sub:
            tl.store(Y[kb][sub], qf.k_superblock(w.q[sub], w.sc[sub], w.d, w.m[sub], w.dmin))
