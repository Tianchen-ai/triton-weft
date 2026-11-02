from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_iq1_m(
    W: tl.View[ggml_formats.IQ1_M, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[ggml_formats.I8X8, (2048, 8)],
    f16_bits: tl.View[tl.f32, (65536,)],
    Y: tl.View[tl.f32, (M, N)],
 *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    grid_values = grid.values
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

                        sc0 = tl.u32(w.scales[:, 0]) | tl.u32(w.scales[:, 1]) << tl.u32(8)
                        sc1 = tl.u32(w.scales[:, 2]) | tl.u32(w.scales[:, 3]) << tl.u32(8)
                        sc2 = tl.u32(w.scales[:, 4]) | tl.u32(w.scales[:, 5]) << tl.u32(8)
                        sc3 = tl.u32(w.scales[:, 6]) | tl.u32(w.scales[:, 7]) << tl.u32(8)
                        scale_bits = sc0 >> tl.u32(12) | sc1 >> tl.u32(8) & tl.u32(240)
                        scale_bits = scale_bits | sc2 >> tl.u32(4) & tl.u32(3840)
                        scale_bits = scale_bits | sc3 & tl.u32(61440)
                        block_scale = qf.nonlinear_lookup(f16_bits, scale_bits)

                        group = tl.arange(0, 8, dtype=tl.u32, axis="group")
                        scale_part = tl.arange(0, 2, dtype=tl.u32, axis="scale_part")
                        entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
                        payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
                        qh = tl.widen(w.qh[:, group * tl.u32(2) + scale_part], tl.u32)
                        grid_index = tl.widen(
                            w.q[
                                :,
                                group * tl.u32(4)
                                + scale_part * tl.u32(2)
                                + entry,
                            ],
                            tl.u32,
                        ) | (qh >> entry * tl.u32(4) & tl.u32(7)) << tl.u32(8)
                        weight = tl.lookup(
                            grid_values,
                            grid_index * tl.u32(8) + payload,
                            bounds="in_bounds",
                        )
                        activation = x.q[
                            :,
                            group * tl.u32(32)
                            + scale_part * tl.u32(16)
                            + entry * tl.u32(8)
                            + payload,
                        ]
                        dot = tl.reduce_dot(activation, weight, over=("entry", "payload"), acc_dtype=tl.i32)
                        scale_word = tl.widen(
                            w.scales[:, group // tl.u32(2) * tl.u32(2)], tl.u32
                        ) | tl.widen(
                            w.scales[
                                :,
                                group // tl.u32(2) * tl.u32(2) + tl.u32(1),
                            ],
                            tl.u32,
                        ) << tl.u32(8)
                        scale_shift = group % tl.u32(2) * tl.u32(6) + scale_part * tl.u32(3)
                        scale = tl.i32(
                            (scale_word >> scale_shift & tl.u32(7)) * tl.u32(2)
                            + tl.u32(1)
                        )
                        main = tl.reduce(
                            tl.reduce(dot * scale, axis="scale_part"), axis="group"
                        )

                        delta_shift = tl.u32(3) + entry * tl.u32(4)
                        delta_bit_u16 = tl.narrow(
                            qh >> delta_shift & tl.u32(1),
                            tl.u16,
                            rounding="rtz",
                            saturation=False,
                        )
                        delta_bit = tl.narrow(
                            delta_bit_u16,
                            tl.u8,
                            rounding="rtz",
                            saturation=False,
                        )
                        delta = tl.i8(1) - tl.i8(delta_bit) * tl.i8(2)
                        payload_u16 = tl.narrow(
                            payload, tl.u16, rounding="rtz", saturation=False
                        )
                        payload_u8 = tl.narrow(
                            payload_u16,
                            tl.u8,
                            rounding="rtz",
                            saturation=False,
                        )
                        delta = delta + tl.i8(payload_u8) * tl.i8(0)
                        correction_part = tl.reduce_dot(
                            activation,
                            delta,
                            over=("entry", "payload"),
                            acc_dtype=tl.i32,
                        )
                        correction = tl.reduce(
                            tl.reduce(correction_part * scale, axis="scale_part"),
                            axis="group",
                        )
                        combined = tl.f32(main) + tl.f32(0.125) * tl.f32(correction)
                        acc += tl.f32(block_scale) * tl.f32(x.ds) * combined
                    tl.store(Y[mb, nb], acc)
