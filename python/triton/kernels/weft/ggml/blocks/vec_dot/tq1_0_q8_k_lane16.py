from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.TQ1_0, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    powers: tl.View[tl.u32, (5,)],
):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        lane = tl.arange(0, 16, dtype=tl.u32, axis="k")
        tail = qf.radix3_digit_i8(
            powers, w.qh[lane % tl.u32(4)], lane // tl.u32(4)
        )
        partial = tl.widen(x.q[tl.u32(240) + lane], tl.i16) * tl.widen(
            tail, tl.i16
        )
        packed0 = w.q[lane]
        packed1 = w.q[tl.u32(16) + lane]
        packed2 = w.q[tl.u32(32) + lane]
        # Each lane sums 1 + 5 * 3 products of magnitude at most 128: 2048.
        for digit in range(5):
            q0 = qf.radix3_digit_i8(powers, packed0, tl.u32(digit))
            q1 = qf.radix3_digit_i8(powers, packed1, tl.u32(digit))
            q2 = qf.radix3_digit_i8(powers, packed2, tl.u32(digit))
            partial += tl.widen(
                x.q[tl.u32(digit) * tl.u32(32) + lane], tl.i16
            ) * tl.widen(q0, tl.i16)
            partial += tl.widen(
                x.q[tl.u32(digit) * tl.u32(32) + tl.u32(16) + lane], tl.i16
            ) * tl.widen(q1, tl.i16)
            partial += tl.widen(
                x.q[tl.u32(160) + tl.u32(digit) * tl.u32(16) + lane], tl.i16
            ) * tl.widen(q2, tl.i16)
        integer = tl.reduce(tl.widen(partial, tl.i32), axis="k")
        result += tl.f32(integer) * (tl.f32(w.d) * tl.f32(x.ds))
    return result


@triton.jit(model="weft")
def quantized_vec_dot_tq1_0_q8_k_lane16(
    W: tl.View[ggml_formats.TQ1_0, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    powers: tl.View[tl.u32, (5,)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, powers))
