from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit
def compute(W: tl.View[ggml_formats.Q2_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)]):
    lane = tl.arange(0, 16, dtype=tl.u16, axis="k")
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        integer = tl.i32(0)
        for half in tl.static_range(4):
            partial = tl.i16(lane * tl.u16(0))
            for part in tl.static_range(4):
                index = lane + tl.u16(half * 64 + part * 16)
                scale = tl.i8(w.scales[half * 4 + part] & tl.u8(15))
                scaled = tl.i8(w.q[index]) * scale
                # Four products per lane: |sum| <= 4 * 3 * 128 * 15 = 23040.
                partial += tl.i16(scaled) * tl.i16(x.q[index])
            integer += tl.reduce(tl.i32(partial), axis="k")
        minima = tl.i16(tl.u8(w.scales) >> tl.u8(4))
        correction = tl.reduce_dot(x.bsum, minima, over="k", acc_dtype=tl.i32)
        result += tl.f32(x.ds) * (tl.f32(w.d) * tl.f32(integer) - tl.f32(w.dmin) * tl.f32(correction))
    return result


@triton.jit(model="weft")
def dot(W: tl.View[ggml_formats.Q2_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute(W, X))
