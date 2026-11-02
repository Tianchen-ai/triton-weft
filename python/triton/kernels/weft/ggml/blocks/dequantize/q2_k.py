from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_q2_k(W: tl.View[ggml_formats.Q2_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        with tl.level.subtiles(kb, extent=16) as sub:
            metadata = w.scales[sub]
            scale = qf.extract_bits(metadata, 0, 4)
            minimum = qf.extract_bits(metadata, 4, 4)
            tl.store(Y[kb][sub], qf.k_superblock(w.q[sub], scale, w.d, minimum, w.dmin))
