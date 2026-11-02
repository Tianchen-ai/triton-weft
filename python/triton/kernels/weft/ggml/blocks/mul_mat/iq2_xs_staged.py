from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def _iq2_xs_group_products(w, x, grid, signs, scale_group, entry, payload, group):
    group_u32 = tl.u32(group)
    linear_entry = scale_group * tl.u32(2) + entry
    code = tl.widen(w.q[:, group_u32 * tl.u32(4) + linear_entry], tl.u32)
    grid_index = code & tl.u32(511)
    sign_index = code >> tl.u32(9)
    weight = tl.lookup(grid, grid_index * tl.u32(8) + payload, bounds="in_bounds")
    sign = tl.lookup(signs, sign_index * tl.u32(8) + payload, bounds="in_bounds")
    signed_weight = weight * sign
    entry_offset = scale_group * tl.u32(16) + entry * tl.u32(8)
    activation = x.q[:, group_u32 * tl.u32(32) + entry_offset + payload]
    partial = tl.reduce_dot(activation, signed_weight, over=("entry", "payload"), acc_dtype=tl.i32)
    metadata = tl.widen(w.scales[:, group_u32 + scale_group // tl.u32(2)], tl.u32)
    scale = tl.i32(
        (metadata >> scale_group % tl.u32(2) * tl.u32(4) & tl.u32(15)) * tl.u32(2)
        + tl.u32(1)
    )
    return tl.reduce(partial * scale, axis="scale_group")


@triton.jit(model="weft")
def production_mul_mat_iq2_xs_staged(
    W: tl.View[ggml_formats.IQ2_XS, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[tl.i8, (4096,)],
    signs: tl.View[tl.i8, (1024,)],
    Y: tl.View[tl.f32, (M, N)],
 *, KC: tl.constexpr, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
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
                            scale_group = tl.arange(0, 2, dtype=tl.u32, axis="scale_group")
                            entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
                            payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
                            block_sum = tl.state(tl.i32, [MR, NR], init=tl.i32(0))
                            group_index = tl.index(0)
                            with tl.level.subtiles(kb, extent=32) as group:
                                block_sum += _iq2_xs_group_products(
                                    w,
                                    x,
                                    grid,
                                    signs,
                                    scale_group,
                                    entry,
                                    payload,
                                    group_index,
                                )
                                group_index += tl.index(1)
                            f32_acc += (
                                tl.f32(0.125)
                                * tl.f32(w.d)
                                * tl.f32(x.ds)
                                * tl.widen(block_sum, tl.f32)
                            )
                        tl.store(Y[mb, nb], f32_acc)
