from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(W: tl.View[ggml_formats.TQ2_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)]):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        q = tl.i8(w.q) - tl.i8(1)
        integer = tl.reduce_dot(x.q, q, over="k", acc_dtype=tl.i32)
        scale = tl.f32(x.ds) * tl.f32(w.d)
        result += tl.widen(integer, tl.f32) * scale
    return result


@triton.jit(model="weft")
def quantized_vec_dot_tq2_0_q8_k(
    W: tl.View[ggml_formats.TQ2_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]
):
    tl.store(Y[0], compute(W, X))
