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
    power_table = tl.stage(tl.load(powers))
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        lane0 = tl.arange(0, 32, axis="k")
        digit0 = tl.arange(0, 4, dtype=tl.u32, axis="digit0")
        q0 = qf.radix3_digit_i8(power_table, w.q[lane0], digit0)
        partial0 = tl.reduce_dot(x.q[digit0 * tl.u32(32) + lane0], q0, over="k", acc_dtype=tl.i32)
        integer = tl.reduce(partial0, axis="digit0")
        q0_tail = qf.radix3_digit_i8(power_table, w.q[lane0], tl.u32(4))
        integer += tl.reduce_dot(x.q[tl.u32(128) + lane0], q0_tail, over="k", acc_dtype=tl.i32)
        lane1 = tl.arange(0, 16, axis="k")
        digit1 = tl.arange(0, 5, dtype=tl.u32, axis="digit1")
        q1 = qf.radix3_digit_i8(power_table, w.q[tl.u32(32) + lane1], digit1)
        partial1 = tl.reduce_dot(
            x.q[tl.u32(160) + digit1 * tl.u32(16) + lane1],
            q1,
            over="k",
            acc_dtype=tl.i32,
        )
        integer += tl.reduce(partial1, axis="digit1")
        lane2 = tl.arange(0, 4, axis="k")
        digit2 = tl.arange(0, 4, dtype=tl.u32, axis="digit2")
        q2 = qf.radix3_digit_i8(power_table, w.qh[lane2], digit2)
        partial2 = tl.reduce_dot(
            x.q[tl.u32(240) + digit2 * tl.u32(4) + lane2],
            q2,
            over="k",
            acc_dtype=tl.i32,
        )
        integer += tl.reduce(partial2, axis="digit2")
        result += tl.f32(integer) * (tl.f32(w.d) * tl.f32(x.ds))
    return result


@triton.jit(model="weft")
def quantized_vec_dot_tq1_0_q8_k(
    W: tl.View[ggml_formats.TQ1_0, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    powers: tl.View[tl.u32, (5,)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, powers))
