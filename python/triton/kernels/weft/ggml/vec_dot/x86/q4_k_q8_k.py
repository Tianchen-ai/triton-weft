from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from .fields import select

MIN_GROUPS = tl.constexpr(8)


@triton.jit
def compute(W, X: tl.View[ggml_formats.Q8_K, (K,)], LANES: tl.constexpr, ROWS: tl.constexpr = 0,
            point=None, HIGH: tl.constexpr = False):
    """Scaled pair partials; grouped callers pass their row-domain point."""
    tl.static_assert(LANES == 8 or LANES == 16)
    if ROWS:
        acc = tl.state(tl.f32, [ROWS, LANES], init=0)
        minimum = tl.state(tl.f32, [ROWS, MIN_GROUPS], init=0)
    else:
        acc = tl.state(tl.f32, [LANES], init=0)
        minimum = tl.state(tl.f32, [MIN_GROUPS], init=0)
    lane = tl.arange(0, LANES * 4, dtype=tl.u32, axis="LANES")
    pair = tl.arange(0, LANES * 2, dtype=tl.u32, axis="LANES")
    minimum_pair = tl.arange(0, MIN_GROUPS * 2, dtype=tl.u32, axis="MIN_GROUPS")
    with tl.level.blocks(K, extent=256) as kb:
        if ROWS:
            integer = tl.state(tl.i32, [ROWS, LANES], init=0)
            w = W[point, kb]
        else:
            integer = tl.state(tl.i32, [LANES], init=0)
            w = W[kb]
        x = tl.load(X[kb])
        quantized, scales, minima = tl.load(w.q), tl.load(w.sc), tl.load(w.m)
        if HIGH:
            high = tl.load(w.qh)
        for g in tl.static_range(64 // LANES):
            coordinate = tl.u32(g * LANES * 4) + lane
            weights = select(quantized, coordinate, ROWS)
            if HIGH:
                weights = tl.u6(tl.u8(weights) | (tl.u8(select(high, coordinate, ROWS)) << 4))
            products = tl.mac_pairs(weights, x.q[coordinate], into=tl.i16)
            scale = tl.i16(select(scales, tl.u32(g * LANES // 8) + pair // tl.u32(16), ROWS))
            integer += tl.mac_pairs(products, scale, into=tl.i32)
        acc += (tl.f32(tl.load(w.d)) * tl.f32(x.ds)) * tl.f32(integer)
        minimum_terms = tl.mac_pairs(x.bsum[minimum_pair],
                                    tl.i16(select(minima, minimum_pair // tl.u32(2), ROWS)), into=tl.i32)
        minimum += (tl.f32(tl.load(w.dmin)) * tl.f32(x.ds)) * tl.f32(minimum_terms)
    return tl.sum(acc, 1 if ROWS else 0) - tl.sum(minimum, 1 if ROWS else 0)


@triton.jit(model="weft")
def dot(W: tl.View[ggml_formats.Q4_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)],
        *, LANES: tl.constexpr):
    tl.store(Y[0], compute(W, X, LANES))
