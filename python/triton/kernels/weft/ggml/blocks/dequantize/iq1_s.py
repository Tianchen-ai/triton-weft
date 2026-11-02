from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_iq1_s(
    W: tl.View[ggml_formats.IQ1_S, (K,)],
    grid: tl.View[tl.i8, (16384,)],
    Y: tl.View[tl.f32, (K,)],
):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        group = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as group_point:
            metadata = w.qh[group]
            delta = tl.f32(0.125)
            if qf.extract_bits(metadata, 15) != tl.u32(0):
                delta = tl.f32(-0.125)
            scale = tl.f32(w.d) * tl.f32(
                tl.i32(qf.extract_bits(metadata, 12, 3)) * tl.i32(2) + tl.i32(1)
            )
            entry = tl.index(0)
            with tl.level.subtiles(group_point, extent=8) as entry_point:
                lane = tl.arange(0, 8, dtype=tl.u16, axis="k")
                grid_index = tl.u32(w.q[group * tl.index(4) + entry]) | qf.extract_bits(
                    metadata, tl.u32(entry) * tl.u32(3), 3
                ) << tl.u32(8)
                grid_value = tl.lookup(
                    grid, grid_index * tl.u32(8) + tl.u32(lane), bounds="in_bounds"
                )
                tl.store(
                    Y[kb][group_point][entry_point],
                    qf.iq_codebook(tl.f32(tl.i32(grid_value)) + delta, scale),
                )
                entry += tl.index(1)
            group += tl.index(1)
