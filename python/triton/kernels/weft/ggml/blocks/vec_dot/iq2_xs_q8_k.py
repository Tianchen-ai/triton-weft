from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def quantized_vec_dot_iq2_xs_q8_k(
    W: tl.View[ggml_formats.IQ2_XS, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (512, 8)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    Y: tl.View[tl.f32, (1,)],
):
    sumf = tl.f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        scale_group = tl.arange(0, 16, dtype=tl.u32, axis="scale_group")
        entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
        linear_entry = scale_group * tl.u32(2) + entry
        entry_offset = scale_group * tl.u32(16) + entry * tl.u32(8)
        block_sum = qf.iq2_xs_entry_reduce(
            w.q[linear_entry],
            w.scales[scale_group // tl.u32(2)],
            x.q[entry_offset + tl.u32(payload)],
            grid_values,
            sign_values,
            scale_group,
            payload,
        )
        sumf += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(block_sum)
    tl.store(Y[0], tl.f32(0.125) * sumf)
