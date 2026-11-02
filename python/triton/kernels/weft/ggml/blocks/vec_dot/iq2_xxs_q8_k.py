from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.IQ2_XXS, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (256, 8)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
):
    sumf = tl.f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        group = tl.arange(0, 8, dtype=tl.u16, axis="group")
        entry = tl.arange(0, 4, dtype=tl.u16, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
        group_word = group * tl.u16(4)
        word0 = tl.u16(w.q[group_word + (entry >> tl.u16(1))])
        grid_index = word0 >> ((entry & tl.u16(1)) * tl.u16(8)) & tl.u16(255)
        word1 = tl.widen(w.q[group_word + tl.u16(2)], tl.u32) | tl.widen(
            w.q[group_word + tl.u16(3)], tl.u32
        ) << tl.u32(16)
        sign_index = tl.narrow(
            word1 >> tl.u32(entry) * tl.u32(7), tl.u16, rounding="rtz", saturation=False
        ) & tl.u16(127)
        weight = tl.lookup(
            grid_values, grid_index * tl.u16(8) + payload, bounds="in_bounds"
        )
        sign = tl.lookup(
            sign_values, sign_index * tl.u16(8) + payload, bounds="in_bounds"
        )
        activation = x.q[group * tl.u16(32) + entry * tl.u16(8) + payload]
        partial = tl.reduce_dot(activation, weight * sign, over=("entry", "payload"), acc_dtype=tl.i32)
        scale = tl.i32((word1 >> tl.u32(28)) * tl.u32(2) + tl.u32(1))
        block_sum = tl.reduce(partial * scale, axis="group")
        sumf += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(block_sum)
    return tl.f32(0.125) * sumf


@triton.jit(model="weft")
def quantized_vec_dot_iq2_xxs_q8_k(
    W: tl.View[ggml_formats.IQ2_XXS, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (256, 8)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, grid, signs))
