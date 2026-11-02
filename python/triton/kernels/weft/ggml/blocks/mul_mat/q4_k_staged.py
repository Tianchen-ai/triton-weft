from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_q4_k_staged(
    W: tl.View[ggml_formats.Q4_K, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
 *, KC: tl.constexpr, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], tl.f32(0.0))
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        with tl.level.tiles(K, extent=tl.auto("KC")) as kc:
            wp = tl.stage(tl.load(W[nc, kc]))
            with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
                with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                    with tl.level.cols(nc, group=16) as nb:
                        f32_acc = tl.state(tl.f32, [MR, 16], init=tl.load(Y[mb, nb]))
                        with tl.level.blocks(kc, extent=256) as kb:
                            w = wp[nb, kb]
                            x = tl.load(Xq[mb, kb])
                            i32_acc = tl.state(tl.i32, [MR, 16], init=0)
                            with tl.level.subtiles(extent=32) as s:
                                p16 = tl.mac_groups(x.q[s], w.q[s], n=4, into=tl.i16)
                                i32_acc += tl.reduce(tl.widen(p16, tl.i32)) * w.sc[s]
                            mins = tl.sum_pairs(x.bsum)
                            min_term = tl.dot(mins, w.m, over="k", acc_dtype=tl.i32)
                            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
                        tl.store(Y[mb, nb], f32_acc)
