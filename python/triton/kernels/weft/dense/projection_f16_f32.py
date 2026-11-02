from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def projection_f16_f32(
    W: tl.View[tl.f16, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
    *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Project F32 activations through F16 weights with F32 accumulation."""
    with tl.level.tiles(M, extent=BM) as mb:
        with tl.level.tiles(N, extent=BN) as nb:
            acc = tl.state(tl.f32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=BK) as kb:
                x = tl.load(X[mb, kb])
                w = tl.widen(tl.load(W[nb, kb]), tl.f32)
                acc += tl.dot(x, w, over="k", acc_dtype=tl.f32)
            tl.store(Y[mb, nb], acc)
