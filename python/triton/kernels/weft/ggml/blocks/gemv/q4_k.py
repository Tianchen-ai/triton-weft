from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def q4_k_q8_k_gemv(
    W: tl.View[ggml_formats.Q4K_I[16], (M, K)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    Y: tl.View[tl.f32, (M,)],
):
    with tl.level.rows(M, group=16) as mb:
        f32_acc = tl.state(tl.f32, [16], init=0)
        with tl.level.blocks(K, extent=256) as kb:
            w = tl.load(W[mb, kb])
            x = tl.load(X[kb])
            i32_acc = tl.state(tl.i32, [16], init=0)
            with tl.level.subtiles(extent=32) as s:
                p16 = tl.mac_pairs(w.q[s], x.q[s], into=tl.i16)
                i32_acc += tl.reduce(tl.widen(p16, tl.i32)) * w.sc[s]
            mins = tl.sum_pairs(x.bsum)
            min_term = tl.reduce_dot(w.m, mins, over="k")
            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
        tl.store(Y[mb], f32_acc)
