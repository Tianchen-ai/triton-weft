from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.MXFP4, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    e8m0_scale: tl.View[tl.f32, (256,)],
):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=32) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        integer = qf.dot_codebook32(w.q, x.q, codebook)
        scale = tl.f32(x.d) * qf.exponent_scale(e8m0_scale, w.e)
        result += scale * tl.f32(integer)
    return result


@triton.jit(model="weft")
def quantized_vec_dot_mxfp4_q8_0(
    W: tl.View[ggml_formats.MXFP4, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    scale: tl.View[tl.f32, (256,)],
    Y: tl.View[tl.f32, (1,)],
):
    tl.store(Y[0], compute(W, X, codebook, scale))
