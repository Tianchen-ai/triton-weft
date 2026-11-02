from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(noinline=True)
def multiply_quantized(
    W: tl.View[ggml_formats.Q2_K, (N, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
 *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
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
                        low_scales = tl.stage(
                            tl.u8(wp[nb, kb].scales) & tl.u8(15)
                        )
                        integer = tl.state(tl.i32, [MR, NR], init=tl.i32(0))
                        with tl.level.subtiles(kb, extent=16) as sub:
                            # Q2 values are in [0, 3], so the signed byte
                            # operand preserves the exact integer program.
                            partial = tl.dot(x.q[:, sub], tl.widen(w.q[:, sub], tl.i8),
                                             over="k", acc_dtype=tl.i32)
                            scale = tl.i32(low_scales[:, sub])
                            integer += partial * scale
                        mins = tl.widen(tl.u8(w.scales) >> tl.u8(4), tl.i16)
                        correction = tl.dot(x.bsum, mins, over="k", acc_dtype=tl.i32)
                        acc += tl.f32(x.ds) * (
                            tl.f32(w.d) * tl.widen(integer, tl.f32)
                            - tl.f32(w.dmin) * tl.widen(correction, tl.f32)
                        )
                    tl.store(Y[mb, nb], acc)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_q2_k(
    W: tl.View[ggml_formats.Q2_K, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
 *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    multiply_quantized(W, Xq, Y, MC=MC, MR=MR, NC=NC, NR=NR)
