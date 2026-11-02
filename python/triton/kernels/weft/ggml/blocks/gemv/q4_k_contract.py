from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def q4_k_q8_k_gemv_contract(
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
                p32 = tl.reduce_dot(w.q[s], x.q[s], over="k", acc_dtype=tl.i32)
                i32_acc += p32 * w.sc[s]
            mins = tl.sum_pairs(x.bsum)
            min_term = tl.reduce_dot(w.m, mins, over="k")
            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
        tl.store(Y[mb], f32_acc)
