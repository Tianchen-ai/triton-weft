from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def _iq2_xxs_group_products(
    w, x, grid, signs, entry_lane, codebook_lane, group, word_indices, activation_indices
):
    group_word = group * 4
    indices = tl.widen(w.q[:, word_indices], tl.u32)
    grid_index = indices >> tl.u32((entry_lane % 2) * 8) & tl.u32(255)
    word1 = tl.widen(w.q[:, group_word + 2], tl.u32) | tl.widen(
        w.q[:, group_word + 3], tl.u32
    ) << tl.u32(16)
    sign_index = word1 >> tl.u32(entry_lane * 7) & tl.u32(127)
    weight = tl.lookup(grid, grid_index * tl.u32(8) + codebook_lane, bounds="in_bounds")
    sign = tl.lookup(signs, sign_index * tl.u32(8) + codebook_lane, bounds="in_bounds")
    signed_weight = weight * sign
    activation = x.q[:, activation_indices]
    integer = tl.reduce_dot(activation, signed_weight, over=("entry", "payload"), acc_dtype=tl.i32)
    scale = tl.i32((word1 >> tl.u32(28)) * tl.u32(2) + tl.u32(1))
    return integer * scale


@triton.jit(model="weft")
def _compute(
    W: tl.View[ggml_formats.IQ2_XXS, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[ggml_formats.I8X8, (256, 8)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    Y: tl.View[tl.f32, (M, N)],
    bounded_indices: tl.constexpr,
    NC: tl.constexpr, KC: tl.constexpr, MC: tl.constexpr, MR: tl.constexpr, NR: tl.constexpr,
):
    grid_values = grid.values
    sign_values = signs.values
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], tl.f32(0.0))
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        with tl.level.tiles(K, extent=tl.auto("KC")) as kc:
            wp = tl.stage(tl.load(W[nc, kc]))
            with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
                with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                    with tl.level.cols(nc, group=tl.auto("NR")) as nb:
                        f32_acc = tl.state(tl.f32, [MR, NR], init=tl.load(Y[mb, nb]))
                        with tl.level.blocks(kc, extent=256) as kb:
                            w = wp[nb, kb]
                            x = tl.load(Xq[mb, kb])
                            entry_lane = tl.arange(0, 4, axis="entry")
                            codebook_lane = tl.arange(0, 8, axis="payload")
                            entry_u32 = tl.arange(0, 4, dtype=tl.u32, axis="entry")
                            payload_u32 = tl.arange(0, 8, dtype=tl.u32, axis="payload")
                            block_sum = tl.state(tl.i32, [MR, NR], init=tl.i32(0))
                            group_index = tl.index(0)
                            with tl.level.subtiles(kb, extent=32) as group:
                                if bounded_indices:
                                    group_partial = _iq2_xxs_group_products(
                                        w, x, grid_values, sign_values, entry_u32, payload_u32,
                                        group_index,
                                        tl.u32(group_index) * tl.u32(4) + entry_u32 // tl.u32(2),
                                        tl.u32(group_index) * tl.u32(32) + entry_u32 * tl.u32(8) + payload_u32,
                                    )
                                else:
                                    group_partial = _iq2_xxs_group_products(
                                        w, x, grid_values, sign_values, entry_lane, codebook_lane,
                                        group_index,
                                        group_index * 4 + entry_lane // 2,
                                        group_index * 32 + entry_lane * 8 + codebook_lane,
                                    )
                                block_sum += group_partial
                                group_index += tl.index(1)
                            f32_acc += (
                                tl.f32(w.d) * tl.f32(x.ds) * tl.widen(block_sum, tl.f32)
                            )
                        tl.store(Y[mb, nb], f32_acc)
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
            with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                with tl.level.cols(nc, group=tl.auto("NR")) as nb:
                    value = tl.load(Y[mb, nb])
                    tl.store(Y[mb, nb], tl.f32(0.125) * value)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq2_xxs_staged(
    W: tl.View[ggml_formats.IQ2_XXS, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[ggml_formats.I8X8, (256, 8)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    Y: tl.View[tl.f32, (M, N)],
    *, NC: tl.constexpr, KC: tl.constexpr, MC: tl.constexpr, MR: tl.constexpr, NR: tl.constexpr,
):
    _compute(W, X, Xq, grid, signs, Y, False, NC, KC, MC, MR, NR)
