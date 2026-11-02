from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_tq1_0(
    W: tl.View[ggml_formats.TQ1_0, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    powers: tl.View[tl.u32, (5,)],
    Y: tl.View[tl.f32, (M, N)],
 *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    power_table = tl.stage(tl.load(powers))
    quant_q8_k.quantize_matrix(X, Xq)
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        wp = tl.stage(tl.load(W[nc]))
        with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
            xp = tl.stage(tl.load(Xq[mc]))
            with tl.level.cols(nc, group=tl.auto("NR")) as nb:
                with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                    acc = tl.state(tl.f32, [MR, NR], init=tl.f32(0.0))
                    with tl.level.blocks(K, extent=256) as kb:
                        w = wp[nb, kb]
                        x = xp[mb, kb]
                        integer = tl.state(tl.i32, [MR, NR], init=tl.i32(0))
                        lane0 = tl.arange(0, 32, axis="k")
                        digit0 = tl.arange(0, 5, dtype=tl.u32, axis="digit0")
                        q0 = qf.radix3_digit_i8(power_table, w.q[:, lane0], digit0)
                        partial0 = tl.reduce_dot(
                            x.q[:, digit0 * tl.u32(32) + lane0],
                            q0,
                            over="k",
                            acc_dtype=tl.i32,
                        )
                        integer += tl.reduce(partial0, axis="digit0")
                        lane1 = tl.arange(0, 16, axis="k")
                        digit1 = tl.arange(0, 5, dtype=tl.u32, axis="digit1")
                        q1 = qf.radix3_digit_i8(
                            power_table, w.q[:, tl.u32(32) + lane1], digit1
                        )
                        partial1 = tl.reduce_dot(
                            x.q[:, tl.u32(160) + digit1 * tl.u32(16) + lane1],
                            q1,
                            over="k",
                            acc_dtype=tl.i32,
                        )
                        integer += tl.reduce(partial1, axis="digit1")
                        lane2 = tl.arange(0, 16, axis="k")
                        q2 = qf.radix3_digit_i8(
                            power_table, w.qh[:, lane2 // tl.u32(4)], lane2 % tl.u32(4)
                        )
                        integer += tl.dot(x.q[
                                :,
                                tl.u32(240)
                                + lane2 % tl.u32(4) * tl.u32(4)
                                + lane2 // tl.u32(4),
                            ], q2, over="k", acc_dtype=tl.i32)
                        acc += tl.f32(w.d) * tl.f32(x.ds) * tl.widen(integer, tl.f32)
                    tl.store(Y[mb, nb], acc)
