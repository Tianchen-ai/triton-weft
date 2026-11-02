from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(W: tl.View[ggml_formats.Q1_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)]):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=128) as wb:
        w = tl.load(W[wb])
        subtotal = tl.f32(0.0)
        with tl.level.subtiles(wb, extent=32) as xb:
            x = tl.load(X[xb])
            centered = tl.i8(w.q[xb]) * tl.i8(2) - tl.i8(1)
            integer = tl.reduce_dot(x.q, centered, over="k", acc_dtype=tl.i32)
            subtotal += tl.f32(x.d) * tl.f32(integer)
        result += tl.f32(w.d) * subtotal
    return result


@triton.jit(model="weft")
def quantized_vec_dot_q1_0_q8_0(
    W: tl.View[ggml_formats.Q1_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)], Y: tl.View[tl.f32, (1,)]
):
    tl.store(Y[0], compute(W, X))
