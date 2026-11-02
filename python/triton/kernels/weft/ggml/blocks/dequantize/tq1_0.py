from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_tq1_0(
    W: tl.View[ggml_formats.TQ1_0, (K,)],
    powers: tl.View[tl.u32, (5,)],
    Y: tl.View[tl.f32, (K,)],
):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        lane0 = tl.arange(0, 32, axis="k")
        digit0 = tl.arange(0, 5, dtype=tl.u32, axis="tq1_digit0")
        q0 = qf.radix3_digit_i8(powers, w.q[lane0], digit0)
        y0 = tl.reshape(
            qf.ternary_radix(q0, w.d),
            shape=(160,),
            axes=("k",),
            order=("tq1_digit0", "k"),
        )
        tl.store(tl.subview(Y[kb], offsets=(0,), extents=(160,)), y0)
        lane1 = tl.arange(0, 16, axis="k")
        digit1 = tl.arange(0, 5, dtype=tl.u32, axis="tq1_digit1")
        q1 = qf.radix3_digit_i8(powers, w.q[tl.u32(32) + lane1], digit1)
        y1 = tl.reshape(
            qf.ternary_radix(q1, w.d),
            shape=(80,),
            axes=("k",),
            order=("tq1_digit1", "k"),
        )
        tl.store(tl.subview(Y[kb], offsets=(160,), extents=(80,)), y1)
        lane2 = tl.arange(0, 4, axis="k")
        digit2 = tl.arange(0, 4, dtype=tl.u32, axis="tq1_digit2")
        q2 = qf.radix3_digit_i8(powers, w.qh[lane2], digit2)
        y2 = tl.reshape(
            qf.ternary_radix(q2, w.d),
            shape=(16,),
            axes=("k",),
            order=("tq1_digit2", "k"),
        )
        tl.store(tl.subview(Y[kb], offsets=(240,), extents=(16,)), y2)
