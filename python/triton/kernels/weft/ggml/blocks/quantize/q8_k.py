from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def quantize_row(X: tl.View[tl.f32, (K,)], Y: tl.View[ggml_formats.Q8_K, (K,)]):
    with tl.level.blocks(K, extent=256) as kb:
        block = tl.load(X[kb])
        maximum = tl.reduce(block, op="max")
        minimum = tl.reduce(block, op="min")
        extreme = minimum
        if tl.abs(maximum) > tl.abs(minimum):
            extreme = maximum
        inverse = tl.f32(0.0)
        d = tl.f32(0.0)
        if extreme != tl.f32(0.0):
            inverse = tl.f32(-127.0) / extreme
            d = tl.f32(1.0) / inverse
        with tl.level.subtiles(kb, extent=64) as chunk:
            group_values = tl.load(X[chunk])
            q = tl.narrow(
                group_values * inverse, tl.i8, rounding="rne", saturation=True
            )
            tl.store(Y[kb].q[chunk], q)
            with tl.level.subtiles(chunk, extent=16) as group:
                bsum = tl.reduce(tl.widen(q[group], tl.i16), op="add")
                tl.store(Y[kb].bsum[group], tl.i16(bsum))
        tl.store(Y[kb].ds, d)


@triton.jit(model="weft")
def quantize_matrix(X: tl.View[tl.f32, (M, K)], Y: tl.View[ggml_formats.Q8_K, (M, K)]):
    for row in range(M):
        quantize_row(X[row], Y[row])


@triton.jit(model="weft")
def q8_K_quantize(X: tl.View[tl.f32, (K,)], Y: tl.View[ggml_formats.Q8_K, (K,)]):
    quantize_row(X, Y)
