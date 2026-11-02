from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.IQ1_M, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (2048, 8)],
    f16_bits: tl.View[tl.f32, (65536,)],
):
    result = tl.f32(0.0)
    grid_values = grid.values
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        sc0 = tl.u32(w.scales[0]) | tl.u32(w.scales[1]) << tl.u32(8)
        sc1 = tl.u32(w.scales[2]) | tl.u32(w.scales[3]) << tl.u32(8)
        sc2 = tl.u32(w.scales[4]) | tl.u32(w.scales[5]) << tl.u32(8)
        sc3 = tl.u32(w.scales[6]) | tl.u32(w.scales[7]) << tl.u32(8)
        scale_bits = sc0 >> tl.u32(12) | sc1 >> tl.u32(8) & tl.u32(240)
        scale_bits = scale_bits | sc2 >> tl.u32(4) & tl.u32(3840)
        scale_bits = scale_bits | sc3 & tl.u32(61440)
        block_scale = qf.nonlinear_lookup(f16_bits, scale_bits)
        group = tl.arange(0, 8, dtype=tl.u16, axis="group")
        scale_part = tl.arange(0, 2, dtype=tl.u16, axis="scale_part")
        entry = tl.arange(0, 2, dtype=tl.u16, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
        qh = tl.widen(w.qh[group * tl.u16(2) + scale_part], tl.u16)
        high_shift = entry * tl.u16(4)
        grid_index = tl.widen(
            w.q[group * tl.u16(4) + scale_part * tl.u16(2) + entry], tl.u16
        ) | (qh >> high_shift & tl.u16(7)) << tl.u16(8)
        weight = tl.lookup(
            grid_values, grid_index * tl.u16(8) + payload, bounds="in_bounds"
        )
        activation = x.q[
            group * tl.u16(32)
            + scale_part * tl.u16(16)
            + entry * tl.u16(8)
            + payload
        ]
        dot = tl.reduce_dot(activation, weight, over=("entry", "payload"), acc_dtype=tl.i32)
        scale_word = tl.widen(
            w.scales[group // tl.u16(2) * tl.u16(2)], tl.u16
        ) | tl.widen(
            w.scales[group // tl.u16(2) * tl.u16(2) + tl.u16(1)], tl.u16
        ) << tl.u16(8)
        scale_shift = group % tl.u16(2) * tl.u16(6) + scale_part * tl.u16(3)
        scale = tl.i32(
            (scale_word >> scale_shift & tl.u16(7)) * tl.u16(2) + tl.u16(1)
        )
        main = tl.reduce(tl.reduce(dot * scale, axis="scale_part"), axis="group")
        delta_shift = tl.u16(3) + entry * tl.u16(4)
        delta_bit = tl.narrow(
            qh >> delta_shift, tl.u8, rounding="rtz", saturation=False
        ) & tl.u8(1)
        delta = tl.i8(1) - tl.i8(delta_bit) * tl.i8(2)
        payload_u8 = tl.narrow(payload, tl.u8, rounding="rtz", saturation=False)
        delta = delta + tl.i8(payload_u8) * tl.i8(0)
        correction_part = tl.reduce_dot(activation, delta, over=("entry", "payload"), acc_dtype=tl.i32)
        correction = tl.reduce(
            tl.reduce(correction_part * scale, axis="scale_part"), axis="group"
        )
        combined = tl.f32(main) + tl.f32(0.125) * tl.f32(correction)
        result += tl.f32(block_scale) * tl.f32(x.ds) * combined
    return result


@triton.jit(model="weft")
def quantized_vec_dot_iq1_m_q8_k(
    W: tl.View[ggml_formats.IQ1_M, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (2048, 8)],
    f16_bits: tl.View[tl.f32, (65536,)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, grid, f16_bits))
