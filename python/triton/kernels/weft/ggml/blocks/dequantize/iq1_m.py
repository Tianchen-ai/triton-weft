from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def _grid_delta(grid, grid_index, lane, delta):
    value = tl.lookup(
        grid,
        tl.u32(grid_index) * tl.u32(8) + tl.u32(lane),
        bounds="in_bounds",
    )
    return tl.f32(tl.i32(value)) + tl.f32(delta)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_iq1_m(
    W: tl.View[ggml_formats.IQ1_M, (K,)],
    grid: tl.View[tl.i8, (16384,)],
    f16_bits: tl.View[tl.f32, (65536,)],
    Y: tl.View[tl.f32, (K,)],
):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        sc0 = tl.u32(w.scales[0]) | tl.u32(w.scales[1]) << tl.u32(8)
        sc1 = tl.u32(w.scales[2]) | tl.u32(w.scales[3]) << tl.u32(8)
        sc2 = tl.u32(w.scales[4]) | tl.u32(w.scales[5]) << tl.u32(8)
        sc3 = tl.u32(w.scales[6]) | tl.u32(w.scales[7]) << tl.u32(8)
        scale_bits = sc0 >> tl.u32(12) | sc1 >> tl.u32(8) & tl.u32(240)
        scale_bits = scale_bits | sc2 >> tl.u32(4) & tl.u32(3840)
        scale_bits = scale_bits | sc3 & tl.u32(61440)
        d = qf.nonlinear_lookup(f16_bits, scale_bits)
        group = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as group_point:
            scale_word = tl.u32(
                w.scales[tl.index(2) * (group // tl.index(2))]
            ) | tl.u32(
                w.scales[tl.index(2) * (group // tl.index(2)) + tl.index(1)]
            ) << tl.u32(
                8
            )
            entry = tl.index(0)
            with tl.level.subtiles(group_point, extent=8) as entry_point:
                lane = tl.arange(0, 8, dtype=tl.u32, axis="k")
                qh = w.qh[group * tl.index(2) + entry // tl.index(2)]
                half = tl.u32(entry % tl.index(2))
                high_shift = half * tl.u32(4)
                delta_shift = tl.u32(3) + high_shift
                grid_index = tl.u32(w.q[group * tl.index(4) + entry]) | qf.extract_bits(
                    qh, high_shift, 3
                ) << tl.u32(8)
                delta = tl.f32(0.125)
                if qf.extract_bits(qh, delta_shift) != tl.u32(0):
                    delta = tl.f32(-0.125)
                subscale_shift = tl.u32(group % tl.index(2)) * tl.u32(6) + tl.u32(
                    entry // tl.index(2)
                ) * tl.u32(3)
                subscale = tl.i32(
                    qf.extract_bits(scale_word, subscale_shift, 3)
                ) * tl.i32(2) + tl.i32(1)
                tl.store(Y[kb][group_point][entry_point], qf.iq_codebook(
                        _grid_delta(grid, grid_index, lane, delta),
                        d * tl.f32(subscale),
                    ))
                entry += tl.index(1)
            group += tl.index(1)
