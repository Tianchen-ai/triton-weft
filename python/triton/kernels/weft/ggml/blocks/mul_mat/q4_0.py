from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_0 as quant_q8_0
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_q4_0(
    W: tl.View[ggml_formats.Q4_0, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_0, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
 *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    quant_q8_0.quantize_matrix(X, Xq)
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        wp = tl.stage(tl.load(W[nc]))
        with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
            xp = tl.stage(tl.load(Xq[mc]))
            with tl.level.cols(nc, group=tl.auto("NR")) as nb:
                with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                    acc = tl.state(tl.f32, [MR, NR], init=tl.f32(0.0))
                    with tl.level.blocks(K, extent=32) as kb:
                        w = wp[nb, kb]
                        x = xp[mb, kb]
                        centered = tl.i8(w.q) - tl.i8(8)
                        integer = tl.dot(x.q, centered, over="k", acc_dtype=tl.i32)
                        acc += tl.f32(w.d) * tl.f32(x.d) * tl.widen(integer, tl.f32)
                    tl.store(Y[mb, nb], acc)
