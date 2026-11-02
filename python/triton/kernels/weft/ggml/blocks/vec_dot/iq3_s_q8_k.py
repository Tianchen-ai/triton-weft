from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.IQ3_S, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X4, (512, 4)],
):
    result = tl.f32(0.0)
    grid_values = grid.values
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        half = tl.arange(0, 2, dtype=tl.u32, axis="half")
        entry = tl.arange(0, 4, dtype=tl.u32, axis="entry")
        code = tl.arange(0, 2, dtype=tl.u32, axis="code")
        payload = tl.arange(0, 4, dtype=tl.u16, axis="payload")
        block_sum = tl.i32(0)
        group_index = tl.index(0)
        with tl.level.subtiles(kb, extent=64) as group:
            storage_coordinate = (
                tl.u32(group_index) * tl.u32(16)
                + half * tl.u32(8)
                + entry * tl.u32(2)
                + code
            )
            grid_index = tl.widen(w.q[storage_coordinate], tl.u32) | tl.widen(
                w.qh[storage_coordinate], tl.u32
            ) << tl.u32(8)
            weight = tl.lookup(
                grid_values, grid_index * tl.u32(4) + payload, bounds="in_bounds"
            )
            sign_bit = w.signs[
                tl.u32(group_index) * tl.u32(64)
                + half * tl.u32(32)
                + entry * tl.u32(8)
                + code * tl.u32(4)
                + tl.u32(payload)
            ]
            signed_weight = weight * (tl.i8(1) - tl.i8(sign_bit) * tl.i8(2))
            activation = x.q[
                tl.u32(group_index) * tl.u32(64)
                + half * tl.u32(32)
                + entry * tl.u32(8)
                + code * tl.u32(4)
                + tl.u32(payload)
            ]
            local = tl.reduce_dot(
                activation,
                signed_weight,
                over=("entry", "code", "payload"),
                acc_dtype=tl.i32,
            )
            scale = tl.i32(w.scales[tl.u32(group_index) * tl.u32(2) + half]) * tl.i32(
                2
            ) + tl.i32(1)
            block_sum += tl.reduce(local * scale, axis="half")
            group_index += tl.index(1)
        result += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(block_sum)
    return result


@triton.jit(model="weft")
def quantized_vec_dot_iq3_s_q8_k(
    W: tl.View[ggml_formats.IQ3_S, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X4, (512, 4)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, grid))
