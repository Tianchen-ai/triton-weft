from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(W: tl.View[ggml_formats.Q2_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)]):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        integer = tl.i32(0)
        with tl.level.subtiles(kb, extent=16) as sub:
            partial = tl.reduce_dot(w.q[sub], x.q[sub], over="k", acc_dtype=tl.i32)
            scale = tl.i32(w.scales[sub]) & tl.i32(15)
            integer += partial * scale
        mins = tl.widen(tl.u8(w.scales) >> tl.u8(4), tl.i16)
        minimum = tl.reduce_dot(x.bsum, mins, over="k", acc_dtype=tl.i32)
        result += tl.f32(x.ds) * (
            tl.f32(w.d) * tl.f32(integer) - tl.f32(w.dmin) * tl.f32(minimum)
        )
    return result


@triton.jit(model="weft")
def quantized_vec_dot_q2_k_q8_k(
    W: tl.View[ggml_formats.Q2_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]
):
    tl.store(Y[0], compute(W, X))
