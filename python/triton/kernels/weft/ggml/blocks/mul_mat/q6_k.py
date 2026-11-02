from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_q6_k(
    W: tl.View[ggml_formats.Q6_K, (N, K)],
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
                        integer_acc = tl.state(tl.i32, [MR, NR], init=tl.i32(0))
                        with tl.level.subtiles(kb, extent=64) as pair:
                            q = tl.i8(w.ql[:, pair]) | tl.i8(w.qh[:, pair]) << tl.u8(4)
                            q -= tl.i8(32)
                            with tl.level.subtiles(pair, extent=16) as sub:
                                integer = tl.dot(x.q[:, sub], q[:, sub], over="k", acc_dtype=tl.i32)
                                integer_acc += integer * tl.i32(w.scales[:, sub])
                        acc += (
                            tl.f32(x.ds) * tl.f32(w.d) * tl.widen(integer_acc, tl.f32)
                        )
                    tl.store(Y[mb, nb], acc)
