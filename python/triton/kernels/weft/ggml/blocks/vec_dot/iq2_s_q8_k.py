from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.IQ2_S, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (1024, 8)],
):
    sumf = tl.f32(0.0)
    grid_values = grid.values
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        scale_group = tl.arange(0, 4, dtype=tl.u32, axis="scale_group")
        entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
        block_sum = tl.i32(0)
        group_index = tl.index(0)
        with tl.level.subtiles(kb, extent=64) as group:
            linear_entry = scale_group * tl.u32(2) + entry
            grid_index = tl.widen(
                w.q[tl.u32(group_index) * tl.u32(8) + linear_entry], tl.u32
            ) | (
                (
                    tl.widen(w.qh[group_index * tl.index(2)], tl.u32)
                    | tl.widen(w.qh[group_index * tl.index(2) + tl.index(1)], tl.u32)
                    << tl.u32(8)
                )
                >> linear_entry * tl.u32(2)
                & tl.u32(3)
            ) << tl.u32(
                8
            )
            sign_bit = w.signs[
                tl.u32(group_index) * tl.u32(64)
                + linear_entry * tl.u32(8)
                + tl.u32(payload)
            ]
            weight = tl.lookup(
                grid_values, grid_index * tl.u32(8) + payload, bounds="in_bounds"
            )
            signed_weight = weight * (tl.i8(1) - tl.i8(sign_bit) * tl.i8(2))
            activation = x.q[
                tl.u32(group_index) * tl.u32(64)
                + linear_entry * tl.u32(8)
                + tl.u32(payload)
            ]
            partial = tl.reduce_dot(activation, signed_weight, over=("entry", "payload"), acc_dtype=tl.i32)
            metadata = tl.widen(
                w.scales[tl.u32(group_index) * tl.u32(2) + scale_group // tl.u32(2)],
                tl.u32,
            )
            scale = tl.i32(
                (metadata >> scale_group % tl.u32(2) * tl.u32(4) & tl.u32(15))
                * tl.u32(2)
                + tl.u32(1)
            )
            block_sum += tl.reduce(partial * scale, axis="scale_group")
            group_index += tl.index(1)
        sumf += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(block_sum)
    return tl.f32(0.125) * sumf


@triton.jit(model="weft")
def quantized_vec_dot_iq2_s_q8_k(
    W: tl.View[ggml_formats.IQ2_S, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X8, (1024, 8)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, grid))
