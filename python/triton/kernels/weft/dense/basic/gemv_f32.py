from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def gemv_f32(
    W: tl.View[tl.f32, (M, K)], X: tl.View[tl.f32, (K,)], Y: tl.View[tl.f32, (M,)]
, *, KB: tl.constexpr, MR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as mb:
        acc = tl.state(tl.f32, [MR], init=0)
        with tl.level.blocks(K, extent=tl.auto("KB")) as kb:
            x = tl.load(X[kb])
            acc += tl.reduce_dot(tl.load(W[mb, kb]), x, over="k")
        tl.store(Y[mb], acc)
