from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
from ..quantize import q8_0 as quant_q8_0
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_mxfp4(
    W: tl.View[ggml_formats.MXFP4, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_0, (M, K)],
    codebook: tl.View[tl.i8, (16,)],
    scale: tl.View[tl.f32, (256,)],
    Y: tl.View[tl.f32, (M, N)],
 *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    table = tl.stage(tl.load(codebook))
    quant_q8_0.quantize_matrix(X, Xq)
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        wp = tl.stage(tl.load(W[nc]))
        with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
            xp = tl.stage(tl.load(Xq[mc]))
            with tl.level.cols(nc, group=tl.auto("NR")) as nb:
                with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                    acc = tl.state(tl.f32, [MR, NR], init=tl.f32(0.0))
                    with tl.level.blocks(K, extent=32) as kb:
                        w = wp[nb, kb]
                        x = xp[mb, kb]
                        q = tl.lookup(table, tl.u8(w.q), bounds="in_bounds")
                        integer = tl.dot(x.q, q, over="k", acc_dtype=tl.i32)
                        block_scale = qf.exponent_scale(scale, w.e)
                        acc += tl.f32(x.d) * block_scale * tl.widen(integer, tl.f32)
                    tl.store(Y[mb, nb], acc)
