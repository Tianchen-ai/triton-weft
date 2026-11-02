from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.IQ3_XXS, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X4, (256, 4)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    INDEX_BITS: tl.constexpr = 32,
):
    tl.static_assert(INDEX_BITS == 16 or INDEX_BITS == 32)
    index_dtype: tl.constexpr = tl.u16 if INDEX_BITS == 16 else tl.u32
    sumf = tl.f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        half = tl.arange(0, 2, dtype=tl.u32, axis="half")
        pair_entry = tl.arange(0, 4, dtype=tl.u32, axis="pair_entry")
        pair = tl.arange(0, 2, dtype=tl.u32, axis="pair")
        entry_code = pair_entry * tl.u32(2) + pair
        payload = tl.arange(0, 4, dtype=tl.u16, axis="payload")
        sign_payload = tl.arange(0, 8, dtype=tl.u16, axis="sign_payload")
        block_sum = tl.i32(0)
        group_index = tl.index(0)
        with tl.level.subtiles(kb, extent=64) as group:
            metadata = w.metadata[tl.u32(group_index) * tl.u32(2) + half]
            grid_index = tl.widen(
                w.q[
                    tl.u32(group_index) * tl.u32(16)
                    + half * tl.u32(8)
                    + entry_code
                ],
                index_dtype,
            )
            sign_index = index_dtype(metadata >> pair_entry * tl.u32(7)) & index_dtype(127)
            weight = tl.lookup(grid_values, grid_index * index_dtype(4) + payload, bounds="in_bounds")
            sign_bytes = tl.lookup(sign_values, sign_index * index_dtype(8) + sign_payload, bounds="in_bounds")
            sign = tl.reshape(
                sign_bytes,
                shape=(2, 4, 2, 4),
                axes=("half", "pair_entry", "pair", "payload"),
                order=("half", "pair_entry", "sign_payload"),
            )
            activation = x.q[
                tl.u32(group_index) * tl.u32(64)
                + half * tl.u32(32)
                + entry_code * tl.u32(4)
                + tl.u32(payload)
            ]
            local = tl.reduce_dot(
                activation,
                weight * sign,
                over=("pair_entry", "pair", "payload"),
                acc_dtype=tl.i32,
            )
            scale = tl.i32((metadata >> tl.u32(28)) * tl.u32(2) + tl.u32(1))
            block_sum += tl.reduce(local * scale, axis="half")
            group_index += tl.index(1)
        sumf += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(block_sum)
    return tl.f32(0.25) * sumf


@triton.jit(model="weft")
def quantized_vec_dot_iq3_xxs_q8_k(
    W: tl.View[ggml_formats.IQ3_XXS, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    grid: tl.View[ggml_formats.I8X4, (256, 4)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    Y: tl.View[tl.f32, (1,)],
    *, INDEX_BITS: tl.constexpr = 32,
):
    tl.store(Y[0], compute(W, X, grid, signs, INDEX_BITS))
