from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_iq4_xs(
    W: tl.View[ggml_formats.IQ4_XS, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    codebook: tl.View[tl.i8, (16,)],
    Y: tl.View[tl.f32, (M, N)],
 *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    table = tl.stage(tl.load(codebook))
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
                        sub_index = tl.index(0)
                        with tl.level.subtiles(kb, extent=32) as sub:
                            low = tl.u32(w.scales_l[:, sub_index // tl.index(2)])
                            shift = sub_index % tl.index(2) * tl.index(4)
                            scale_low = low >> tl.u32(shift) & tl.u32(15)
                            scale_high = tl.u32(w.scales_h) >> tl.u32(
                                sub_index * tl.index(2)
                            ) & tl.u32(3)
                            scale = tl.i32(
                                scale_low | scale_high << tl.u32(4)
                            ) - tl.i32(32)
                            q = tl.lookup(table, tl.u8(w.q[:, sub]), bounds="in_bounds")
                            integer = tl.dot(x.q[:, sub], q, over="k", acc_dtype=tl.i32)
                            acc += (
                                tl.f32(w.d)
                                * tl.f32(x.ds)
                                * tl.widen(scale, tl.f32)
                                * tl.widen(integer, tl.f32)
                            )
                            sub_index += tl.index(1)
                    tl.store(Y[mb, nb], acc)
