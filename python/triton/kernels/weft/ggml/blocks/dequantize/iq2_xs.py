from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_dequantize_iq2_xs(
    W: tl.View[ggml_formats.IQ2_XS, (K,)],
    grid: tl.View[tl.i8, (4096,)],
    signs: tl.View[tl.i8, (1024,)],
    Y: tl.View[tl.f32, (K,)],
):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.stage(tl.load(W[kb]))
        d = tl.stage(tl.f32(w.d))
        group = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as group_point:
            scale_byte = tl.stage(w.scales[group_point])
            scale_low = tl.stage(
                d
                * (tl.f32(0.5) + tl.f32(qf.extract_bits(scale_byte, 0, 4)))
                * tl.f32(0.25)
            )
            scale_high = tl.stage(
                d
                * (tl.f32(0.5) + tl.f32(qf.extract_bits(scale_byte, 4, 4)))
                * tl.f32(0.25)
            )
            entry = tl.index(0)
            with tl.level.subtiles(group_point, extent=8) as entry_point:
                lane = tl.arange(0, 8, dtype=tl.u8, axis="k")
                word = w.q[group * tl.index(4) + entry]
                grid_index = tl.narrow(
                    qf.extract_bits(word, 0, 9),
                    tl.u16,
                    rounding="rtz",
                    saturation=False,
                )
                sign_index = tl.narrow(
                    qf.extract_bits(word, 9, 7),
                    tl.u16,
                    rounding="rtz",
                    saturation=False,
                )
                scale = scale_low
                if entry >= tl.index(2):
                    scale = scale_high
                grid_value = tl.lookup(
                    grid, grid_index * tl.u16(8) + tl.u16(lane), bounds="in_bounds"
                )
                sign_value = tl.lookup(
                    signs, sign_index * tl.u16(8) + tl.u16(lane), bounds="in_bounds"
                )
                tl.store(
                    Y[kb][group_point][entry_point],
                    qf.iq_codebook(tl.i32(grid_value) * tl.i32(sign_value), scale),
                )
                entry += tl.index(1)
            group += tl.index(1)
