from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.IQ4_XS, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    codebook: tl.View[tl.i8, (16,)],
):
    table = tl.stage(tl.load(codebook))
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        sub_index = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as sub:
            low = tl.u32(w.scales_l[sub_index // tl.index(2)])
            shift = sub_index % tl.index(2) * tl.index(4)
            scale_low = low >> tl.u32(shift) & tl.u32(15)
            scale_high = tl.u32(w.scales_h) >> tl.u32(sub_index * tl.index(2)) & tl.u32(
                3
            )
            scale = tl.i32(scale_low | scale_high << tl.u32(4)) - tl.i32(32)
            q = qf.small_nonlinear_lookup(table, w.q[sub])
            integer = tl.reduce_dot(q, x.q[sub], over="k", acc_dtype=tl.i32)
            block_scale = tl.f32(w.d) * tl.f32(x.ds) * tl.f32(scale)
            result += block_scale * tl.f32(integer)
            sub_index += tl.index(1)
    return result


@triton.jit(model="weft")
def quantized_vec_dot_iq4_xs_q8_k(
    W: tl.View[ggml_formats.IQ4_XS, (K,)],
    X: tl.View[ggml_formats.Q8_K, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, codebook))
