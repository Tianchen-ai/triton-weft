from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_iq2_s(
    W: tl.View[ggml_formats.IQ2_S, (K,)],
    grid: tl.View[tl.i8, (8192,)],
    Y: tl.View[tl.f32, (K,)],
):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        group = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as group_point:
            high = w.qh[group]
            scale_bits = w.scales[group]
            entry = tl.index(0)
            with tl.level.subtiles(group_point, extent=8) as entry_point:
                lane = tl.arange(0, 8, dtype=tl.u32, axis="k")
                grid_index = tl.u32(w.q[group * tl.index(4) + entry]) | qf.extract_bits(
                    high, tl.u32(entry) * tl.u32(2), 2
                ) << tl.u32(8)
                value = qf.nonlinear_lookup(grid, grid_index * tl.u32(8) + lane)
                sign = tl.i32(1) - tl.i32(w.signs[entry_point]) * tl.i32(2)
                subscale = qf.extract_bits(
                    scale_bits, tl.u32(entry // tl.index(2)) * tl.u32(4), 4
                )
                scale = tl.f32(w.d) * (tl.f32(0.5) + tl.f32(subscale)) * tl.f32(0.25)
                tl.store(Y[kb][group_point][entry_point], qf.iq_codebook(tl.i32(value) * sign, scale))
                entry += tl.index(1)
            group += tl.index(1)
