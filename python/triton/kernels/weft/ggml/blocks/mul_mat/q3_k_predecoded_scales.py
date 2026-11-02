from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_q3_k_predecoded_scales(
    W: tl.View[ggml_formats.Q3_K, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
 *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        wp = tl.stage(tl.load(W[nc]))
        with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
            xp = tl.stage(tl.load(Xq[mc]))
            with tl.level.cols(nc, group=tl.auto("NR")) as nb:
                with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                    acc = tl.state(tl.f32, [MR, NR], init=tl.f32(0.0))
                    with tl.level.blocks(K, extent=256) as kb:
                        w = wp[nb, kb]
                        x = xp[mb, kb]
                        decoded_scales = tl.stage(
                            tl.i8(
                                (
                                    tl.u8(wp[nb, kb].scale_low)
                                    | tl.u8(wp[nb, kb].scale_high) << tl.u8(4)
                                )
                                - tl.u8(32)
                            )
                        )
                        integer_acc = tl.state(tl.i32, [MR, NR], init=tl.i32(0))
                        half_index = tl.index(0)
                        with tl.level.subtiles(kb, extent=128) as half:
                            plane = tl.arange(0, 4, dtype=tl.u8)
                            scale_part = tl.arange(0, 2, dtype=tl.u8)
                            lane = tl.arange(0, 16, dtype=tl.u8, axis="k")
                            coordinate = (
                                tl.u8(half_index) * tl.u8(128)
                                + plane * tl.u8(32)
                                + scale_part * tl.u8(16)
                                + lane
                            )
                            q = tl.i8(
                                tl.u8(w.q[:, coordinate])
                                | tl.u8(w.hmask[:, coordinate]) << tl.u8(2)
                            ) - tl.i8(4)
                            products = tl.reduce_dot(x.q[:, coordinate], q, over="k", acc_dtype=tl.i32)
                            scale_coordinate = (
                                tl.u8(half_index) * tl.u8(8)
                                + plane * tl.u8(2)
                                + scale_part
                            )
                            scale = tl.i32(decoded_scales[:, scale_coordinate])
                            integer_acc += tl.reduce(
                                tl.reduce(products * scale, axis=1), axis=1
                            )
                            half_index += tl.index(1)
                        acc += (
                            tl.f32(x.ds) * tl.f32(w.d) * tl.widen(integer_acc, tl.f32)
                        )
                    tl.store(Y[mb, nb], acc)
