from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.NVFP4, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    ue4m3_scale: tl.View[tl.f32, (256,)],
):
    table = tl.stage(tl.load(codebook))
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=64) as wb:
        w = tl.load(W[wb])
        with tl.level.subtiles(wb, extent=32) as xb:
            x = tl.load(X[xb])
            with tl.level.subtiles(xb, extent=16) as sub:
                q = qf.small_nonlinear_lookup(table, w.q[sub])
                integer = tl.reduce_dot(q, x.q[sub], over="k", acc_dtype=tl.i32)
                scale = tl.f32(x.d) * qf.exponent_scale(ue4m3_scale, w.d[sub])
                result += scale * tl.f32(integer)
    return result


@triton.jit(model="weft")
def quantized_vec_dot_nvfp4_q8_0(
    W: tl.View[ggml_formats.NVFP4, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    scale: tl.View[tl.f32, (256,)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, codebook, scale))
