from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def quantize_row(X: tl.View[tl.f32, (K,)], Y: tl.View[ggml_formats.Q8_0, (K,)]):
    with tl.level.blocks(K, extent=32) as kb:
        x = tl.load(X[kb])
        amax = tl.reduce(tl.abs(x), op="max")
        d = amax / tl.f32(127.0)
        inverse = tl.f32(0.0)
        if amax != tl.f32(0.0):
            inverse = tl.f32(1.0) / d
        q = tl.narrow(x * inverse, tl.i8, rounding="dynamic", saturation=False)
        tl.store(Y[kb].d, tl.f16(d))
        tl.store(Y[kb].q, q)


@triton.jit(model="weft")
def quantize_matrix(X: tl.View[tl.f32, (M, K)], Y: tl.View[ggml_formats.Q8_0, (M, K)]):
    for row in range(M):
        quantize_row(X[row], Y[row])


@triton.jit(model="weft")
def q8_0_quantize(X: tl.View[tl.f32, (K,)], Y: tl.View[ggml_formats.Q8_0, (K,)]):
    quantize_row(X, Y)
