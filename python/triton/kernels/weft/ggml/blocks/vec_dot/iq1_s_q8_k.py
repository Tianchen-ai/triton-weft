from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.IQ1_S, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (2048, 8)],
):
    result = tl.f32(0.0)
    grid_values = grid.values
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        group = tl.arange(0, 8, dtype=tl.u16, axis="group")
        entry = tl.arange(0, 4, dtype=tl.u16, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
        metadata = w.qh[group]
        scale = tl.i32(
            (metadata >> tl.u16(12) & tl.u16(7)) * tl.u16(2) + tl.u16(1)
        )
        grid_index = tl.widen(w.q[group * tl.u16(4) + entry], tl.u16) | (
            metadata >> entry * tl.u16(3) & tl.u16(7)
        ) << tl.u16(8)
        weight = tl.lookup(
            grid_values, grid_index * tl.u16(8) + payload, bounds="in_bounds"
        )
        activation = x.q[
            group * tl.u16(32) + entry * tl.u16(8) + payload
        ]
        group_sum = tl.reduce_dot(activation, weight, over=("entry", "payload"), acc_dtype=tl.i32)
        delta = tl.i32((tl.i16(metadata) >> tl.i16(15)) | tl.i16(1))
        main = tl.reduce(scale * group_sum, axis="group")
        correction = tl.reduce(
            scale
            * delta
            * (
                tl.i32(x.bsum[group * tl.u16(2)])
                + tl.i32(x.bsum[group * tl.u16(2) + tl.u16(1)])
            ),
            axis="group",
        )
        combined = tl.f32(main) + tl.f32(0.125) * tl.f32(correction)
        result += tl.f32(w.d) * tl.f32(x.ds) * combined
    return result


@triton.jit(model="weft")
def quantized_vec_dot_iq1_s_q8_k(
    W: tl.View[ggml_formats.IQ1_S, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (2048, 8)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, grid))
