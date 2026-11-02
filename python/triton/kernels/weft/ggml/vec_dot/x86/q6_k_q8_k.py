from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from .fields import select


@triton.jit
def compute(W, X: tl.View[ggml_formats.Q8_K, (K,)], LANES: tl.constexpr, ROWS: tl.constexpr = 0, point=None):
    """Six-bit pairs and block-sum correction, with an explicit grouped row point."""
    if ROWS:
        acc = tl.state(tl.f32, [ROWS, LANES], init=0)
    else:
        acc = tl.state(tl.f32, [LANES], init=0)
    tl.static_assert(LANES == 8 or LANES == 16)
    lane = tl.arange(0, LANES * 4, dtype=tl.u32, axis="LANES")
    pair = tl.arange(0, LANES * 2, dtype=tl.u32, axis="LANES")
    scale_index = tl.arange(0, 16, dtype=tl.u32, axis="LANES")
    with tl.level.blocks(K, extent=256) as kb:
        if ROWS:
            integer = tl.state(tl.i32, [ROWS, LANES], init=0)
            w = tl.load(W[point, kb])
        else:
            integer = tl.state(tl.i32, [LANES], init=0)
            w = tl.load(W[kb])
        x = tl.load(X[kb])
        for g in tl.static_range(64 // LANES):
            coordinate = tl.u32(g * LANES * 4) + lane
            q = tl.u8(select(w.ql, coordinate, ROWS)) | (tl.u8(select(w.qh, coordinate, ROWS)) << tl.u8(4))
            products = tl.mac_pairs(tl.u6(q), x.q[coordinate], into=tl.i16)
            scales = tl.i16(select(w.scales, tl.u32(g * LANES // 4) + pair // tl.u32(8), ROWS))
            integer += tl.mac_pairs(products, scales, into=tl.i32)
        if LANES == 8:
            correction = tl.mac_pairs(x.bsum[scale_index], tl.i16(select(w.scales, scale_index, ROWS)), into=tl.i32)
        else:
            correction = tl.i32(x.bsum[scale_index]) * tl.i32(select(w.scales, scale_index, ROWS))
        integer -= correction * tl.i32(32)
        acc += (tl.f32(w.d) * tl.f32(x.ds)) * tl.f32(integer)
    return tl.sum(acc, 1 if ROWS else 0)


@triton.jit(model="weft")
def dot(W: tl.View[ggml_formats.Q6_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)],
        *, LANES: tl.constexpr):
    tl.store(Y[0], compute(W, X, LANES))
