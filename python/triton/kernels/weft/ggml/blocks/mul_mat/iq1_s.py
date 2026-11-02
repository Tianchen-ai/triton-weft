from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq1_s(
    W: tl.View[ggml_formats.IQ1_S, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[ggml_formats.I8X8, (2048, 8)],
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
                        entry = tl.arange(0, 4, dtype=tl.u16, axis="entry")
                        payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
                        main = tl.state(tl.i32, [MR, NR], init=tl.i32(0))
                        correction = tl.state(tl.i32, [MR, NR], init=tl.i32(0))
                        group = tl.arange(0, 8, dtype=tl.u16, axis="group")
                        metadata = w.qh[:, group]
                        scale = tl.i32(
                            (metadata >> tl.u16(12) & tl.u16(7)) * tl.u16(2)
                            + tl.u16(1)
                        )
                        grid_index = tl.widen(
                            w.q[:, group * tl.u16(4) + entry], tl.u16
                        ) | (metadata >> entry * tl.u16(3) & tl.u16(7)) << tl.u16(8)
                        weight = tl.lookup(
                            grid_values,
                            grid_index * tl.u16(8) + payload,
                            bounds="in_bounds",
                        )
                        activation = x.q[
                            :,
                            group * tl.u16(32)
                            + entry * tl.u16(8)
                            + payload,
                        ]
                        group_sum = tl.reduce_dot(
                            activation,
                            weight,
                            over=("entry", "payload"),
                            acc_dtype=tl.i32,
                        )
                        main += tl.reduce(scale * group_sum, axis="group")
                        delta = tl.i32(1) - tl.i32(
                            metadata >> tl.u16(15) & tl.u16(1)
                        ) * tl.i32(2)
                        bsum = tl.i32(x.bsum[:, group * tl.u16(2)]) + tl.i32(
                            x.bsum[:, group * tl.u16(2) + tl.u16(1)]
                        )
                        correction += tl.reduce(bsum * scale * delta, axis="group")
                        combined = tl.f32(main) + tl.f32(0.125) * tl.f32(correction)
                        acc += tl.f32(w.d) * tl.f32(x.ds) * combined
                    tl.store(Y[mb, nb], acc)
