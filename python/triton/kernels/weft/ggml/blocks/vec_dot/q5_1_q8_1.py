from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(W: tl.View[ggml_formats.Q5_1, (K,)], X: tl.View[ggml_formats.Q8_1, (K,)]):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=32) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        q = tl.u8(w.q) | tl.u8(w.qh) << tl.u8(4)
        integer = tl.reduce_dot(q, x.q, over="k", acc_dtype=tl.i32)
        result += tl.f32(w.d) * tl.f32(x.d) * tl.f32(integer) + tl.f32(w.m) * tl.f32(
            x.s
        )
    return result


@triton.jit(model="weft")
def quantized_vec_dot_q5_1_q8_1(
    W: tl.View[ggml_formats.Q5_1, (K,)], X: tl.View[ggml_formats.Q8_1, (K,)], Y: tl.View[tl.f32, (1,)]
):
    tl.store(Y[0], compute(W, X))
