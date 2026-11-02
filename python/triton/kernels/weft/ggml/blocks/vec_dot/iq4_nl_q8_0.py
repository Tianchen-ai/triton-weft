from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.IQ4_NL, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    DEFERRED: tl.constexpr = False,
):
    table = tl.stage(tl.load(codebook))
    result = tl.f32(0.0)
    if DEFERRED:
        lane = tl.arange(0, 16, dtype=tl.u16, axis="lane")
        result = tl.f32(lane) * tl.f32(0)
    with tl.level.blocks(K, extent=32) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        if DEFERRED:
            # The fixed GGML IQ4_NL codebook has |entry| <= 127, so the
            # two signed-byte products fit i16: 2 * 127 * 128 = 32512.
            first = tl.lookup(table, tl.u8(w.q[lane]), bounds="in_bounds")
            second = tl.lookup(table, tl.u8(w.q[lane + tl.u16(16)]), bounds="in_bounds")
            integer = tl.i16(first) * tl.i16(x.q[lane])
            integer += tl.i16(second) * tl.i16(x.q[lane + tl.u16(16)])
        else:
            integer = qf.dot_codebook32(w.q, x.q, table)
        result += tl.f32(x.d) * tl.f32(w.d) * tl.f32(integer)
    if DEFERRED:
        return tl.reduce(result, axis="lane")
    return result


@triton.jit(model="weft")
def quantized_vec_dot_iq4_nl_q8_0(
    W: tl.View[ggml_formats.IQ4_NL, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    Y: tl.View[tl.f32, (1,)],
    *, DEFERRED: tl.constexpr = False,
):
    tl.store(Y[0], compute(W, X, codebook, DEFERRED))
