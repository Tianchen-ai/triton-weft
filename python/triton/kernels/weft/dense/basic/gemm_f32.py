from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def gemm_f32(
    X: tl.View[tl.f32, (M, K)], W: tl.View[tl.f32, (N, K)], Y: tl.View[tl.f32, (M, N)]
, *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
            with tl.level.cols(nc, group=tl.auto("NR")) as nb:
                with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                    product = tl.dot(tl.load(X[mb]), tl.load(W[nb]), over="k")
                    tl.store(Y[mb, nb], product)
