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
        entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
        part = tl.arange(0, 2, dtype=tl.u32, axis="scale_part")
        main = tl.i32(entry) * tl.i32(0) + tl.i32(payload) * tl.i32(0)
        correction = main
        # Arbitrary signed-i8 grid values: each lane <= 16 * 16384 * 15.
        # The full i32 sum is bounded by 62914560; correction by 491520.
        for group in range(8):
            qh = tl.widen(w.qh[tl.u32(group) * tl.u32(2) + part], tl.u32)
            index = tl.widen(
                w.q[tl.u32(group) * tl.u32(4) + part * tl.u32(2) + entry],
                tl.u32,
            ) | (qh >> (entry * tl.u32(4)) & tl.u32(7)) << tl.u32(8)
            weight = tl.lookup(
                grid_values, index * tl.u32(8) + payload, bounds="in_bounds"
            )
            activation = x.q[
                tl.u32(group) * tl.u32(32) + part * tl.u32(16)
                + entry * tl.u32(8) + payload
            ]
            scale_word = tl.u32(w.scales[group // 2 * 2]) | tl.u32(
                w.scales[group // 2 * 2 + 1]
            ) << tl.u32(8)
            shift = tl.u32(group % 2) * tl.u32(6) + part * tl.u32(3)
            scale = tl.i16(tl.narrow(
                (scale_word >> shift & tl.u32(7)) * tl.u32(2) + tl.u32(1),
                tl.u16, rounding="rtz", saturation=False,
            ))
            delta16 = tl.narrow(
                qh >> (tl.u32(3) + entry * tl.u32(4)) & tl.u32(1),
                tl.u16, rounding="rtz", saturation=False,
            )
            delta8 = tl.narrow(delta16, tl.u8, rounding="rtz", saturation=False)
            delta = tl.i8(1) - tl.i8(delta8) * tl.i8(2)
            product = tl.widen(activation, tl.i16) * tl.widen(weight, tl.i16)
            correction_product = tl.widen(activation, tl.i16) * tl.widen(delta, tl.i16)
            main += tl.reduce(
                tl.widen(product, tl.i32) * tl.widen(scale, tl.i32),
                axis="scale_part",
            )
            correction += tl.reduce(
                tl.widen(correction_product, tl.i32) * tl.widen(scale, tl.i32),
                axis="scale_part",
            )
        main_lanes = tl.reshape(
            main, shape=(16,), axes=("lane",), order=("entry", "payload")
        )
        correction_lanes = tl.reshape(
            correction, shape=(16,), axes=("lane",), order=("entry", "payload")
        )
        main_sum = tl.reduce(main_lanes, axis="lane")
        correction_sum = tl.reduce(correction_lanes, axis="lane")
        combined = tl.f32(main_sum) + tl.f32(0.125) * tl.f32(correction_sum)
        result += tl.f32(block_scale) * tl.f32(x.ds) * combined
    return result


@triton.jit(model="weft")
def quantized_vec_dot_iq1_m_q8_k_lane16(
    W: tl.View[ggml_formats.IQ1_M, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (2048, 8)],
    f16_bits: tl.View[tl.f32, (65536,)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, grid, f16_bits))
