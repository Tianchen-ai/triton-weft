from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_iq3_s(
    W: tl.View[ggml_formats.IQ3_S, (K,)],
    grid: tl.View[tl.i8, (2048,)],
    Y: tl.View[tl.f32, (K,)],
):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        group = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as group_point:
            subscale = w.scales[group]
            scale = tl.f32(w.d) * tl.f32(tl.i32(1) + tl.i32(2) * tl.i32(subscale))
            pair = tl.index(0)
            with tl.level.subtiles(group_point, extent=8) as pair_point:
                lane = tl.arange(0, 8, dtype=tl.u16, axis="k")
                half = tl.u32(lane // tl.u16(4))
                payload = tl.u32(lane % tl.u16(4))
                storage_coordinate = group * tl.index(8) + pair * tl.index(2)
                grid_index0 = tl.u32(w.q[storage_coordinate]) | tl.u32(
                    w.qh[storage_coordinate]
                ) << tl.u32(8)
                grid_index1 = tl.u32(w.q[storage_coordinate + tl.index(1)]) | tl.u32(
                    w.qh[storage_coordinate + tl.index(1)]
                ) << tl.u32(8)
                grid_index = grid_index0 * (tl.u32(1) - half) + grid_index1 * half
                value = qf.nonlinear_lookup(grid, grid_index * tl.u32(4) + payload)
                sign = tl.i32(1) - tl.i32(w.signs[pair_point]) * tl.i32(2)
                tl.store(Y[kb][group_point][pair_point], qf.iq_codebook(value * sign, scale))
                pair += tl.index(1)
            group += tl.index(1)
