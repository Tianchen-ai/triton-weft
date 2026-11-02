from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(W: tl.View[ggml_formats.Q6_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)]):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        integer_acc = tl.i32(0)
        half_index = tl.index(0)
        with tl.level.subtiles(kb, extent=128) as half:
            plane = tl.arange(0, 4, dtype=tl.u8)
            scale_part = tl.arange(0, 2, dtype=tl.u8)
            lane = tl.arange(0, 16, dtype=tl.u8, axis="k")
            coordinate = (
                tl.u8(half_index) * tl.u8(128)
                + plane * tl.u8(32)
                + scale_part * tl.u8(16)
                + lane
            )
            q = tl.i8(w.ql[coordinate]) | tl.i8(w.qh[coordinate]) << tl.u8(4)
            q -= tl.i8(32)
            products = tl.reduce_dot(x.q[coordinate], q, over="k", acc_dtype=tl.i32)
            scale_coordinate = (
                tl.u8(half_index) * tl.u8(8) + plane * tl.u8(2) + scale_part
            )
            integer_acc += tl.reduce(
                tl.reduce(products * tl.i32(w.scales[scale_coordinate]), axis=1), axis=0
            )
            half_index += tl.index(1)
        result += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(integer_acc)
    return result


@triton.jit(model="weft")
def quantized_vec_dot_q6_k_q8_k(
    W: tl.View[ggml_formats.Q6_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]
):
    tl.store(Y[0], compute(W, X))
