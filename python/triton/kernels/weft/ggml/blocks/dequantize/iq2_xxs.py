from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_iq2_xxs(
    W: tl.View[ggml_formats.IQ2_XXS, (K,)],
    grid: tl.View[tl.i8, (2048,)],
    signs: tl.View[tl.i8, (1024,)],
    Y: tl.View[tl.f32, (K,)],
):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        group = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as group_point:
            group_word = group * tl.index(4)
            metadata = tl.widen(w.q[group_word + tl.index(2)], tl.u32) | tl.widen(
                w.q[group_word + tl.index(3)], tl.u32
            ) << tl.u32(16)
            subscale = tl.f32(0.5) + tl.f32(qf.extract_bits(metadata, 28, 4))
            scale = tl.f32(w.d) * subscale * tl.f32(0.25)
            entry = tl.index(0)
            with tl.level.subtiles(group_point, extent=8) as entry_point:
                lane = tl.arange(0, 8, dtype=tl.u16, axis="k")
                indices = w.q[group_word + entry // tl.index(2)]
                grid_index = tl.narrow(
                    qf.extract_bits(indices, tl.u32(entry % tl.index(2)) * tl.u32(8), 8),
                    tl.u16,
                    rounding="rtz",
                    saturation=False,
                )
                sign_index = tl.narrow(
                    qf.extract_bits(metadata, tl.u32(entry) * tl.u32(7), 7),
                    tl.u16,
                    rounding="rtz",
                    saturation=False,
                )
                grid_value = tl.lookup(
                    grid, grid_index * tl.u16(8) + lane, bounds="in_bounds"
                )
                sign_value = tl.lookup(
                    signs, sign_index * tl.u16(8) + lane, bounds="in_bounds"
                )
                tl.store(
                    Y[kb][group_point][entry_point],
                    qf.iq_codebook(tl.i32(grid_value) * tl.i32(sign_value), scale),
                )
                entry += tl.index(1)
            group += tl.index(1)
