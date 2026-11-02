from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(W: tl.View[ggml_formats.Q5_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)]):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=32) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        q = tl.u8(w.q) | tl.u8(w.qh) << tl.u8(4)
        centered = tl.i8(q) - tl.i8(16)
        integer = tl.reduce_dot(centered, x.q, over="k", acc_dtype=tl.i32)
        result += tl.f32(w.d) * tl.f32(x.d) * tl.f32(integer)
    return result


@triton.jit(model="weft")
def quantized_vec_dot_q5_0_q8_0(
    W: tl.View[ggml_formats.Q5_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)], Y: tl.View[tl.f32, (1,)]
):
    tl.store(Y[0], compute(W, X))
