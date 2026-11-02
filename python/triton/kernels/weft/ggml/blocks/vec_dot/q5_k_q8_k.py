from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(W: tl.View[ggml_formats.Q5_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)]):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        integer = tl.i32(0)
        with tl.level.subtiles(extent=32) as sub:
            q = tl.u8(w.q[sub]) | tl.u8(w.qh[sub]) << tl.u8(4)
            integer += tl.reduce_dot(q, x.q[sub], over="k", acc_dtype=tl.i32) * tl.i32(
                w.sc[sub]
            )
        min_group = tl.arange(0, 8, dtype=tl.u32, axis="min_group")
        minimum = tl.reduce(
            tl.widen(w.m[min_group], tl.i32)
            * (
                tl.widen(x.bsum[min_group * tl.u32(2)], tl.i32)
                + tl.widen(x.bsum[min_group * tl.u32(2) + tl.u32(1)], tl.i32)
            ),
            axis="min_group",
        )
        result += tl.f32(x.ds) * (
            tl.f32(w.d) * tl.f32(integer) - tl.f32(w.dmin) * tl.f32(minimum)
        )
    return result


@triton.jit(model="weft")
def quantized_vec_dot_q5_k_q8_k(
    W: tl.View[ggml_formats.Q5_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]
):
    tl.store(Y[0], compute(W, X))
