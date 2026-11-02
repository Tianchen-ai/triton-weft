from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit
def scale_at(w, index: tl.constexpr):
    return tl.i8((tl.u8(w.scale_low[index]) |
                  (tl.u8(w.scale_high[index]) << tl.u8(4))) - tl.u8(32))


@triton.jit(noinline=True)
def compute_lanes(W: tl.View[ggml_formats.Q3_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                  *, LANES: tl.constexpr = 32):
    """Keep group-scaled integer products and block-scaled float lanes live."""
    tl.static_assert(LANES == 16 or LANES == 32)
    lane = tl.arange(0, LANES, dtype=tl.u32, axis="k")
    result = tl.f32(lane) * tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        x = tl.load(X[block])
        partial = tl.i32(lane) * tl.i32(0)
        for group in tl.static_range(256 // LANES):
            coordinate = lane + tl.u32(group * LANES)
            q = tl.i8(tl.u8(w.q[coordinate]) | (tl.u8(w.hmask[coordinate]) << tl.u8(2))) - tl.i8(4)
            product = tl.i16(q) * tl.i16(x.q[coordinate])
            if LANES == 16:
                scale = scale_at(w, group)
            else:
                scale = tl.where(lane < tl.u32(16), scale_at(w, group * 2), scale_at(w, group * 2 + 1))
            # Each i16 product is exact (magnitude <= 512). Scaling and the
            # at most sixteen contributions use i32, bounded by 262144.
            partial += tl.i32(product) * tl.i32(scale)
        result += tl.f32(partial) * (tl.f32(w.d) * tl.f32(x.ds))
    return tl.reduce(result, axis="k")


@triton.jit(model="weft")
def dot_lanes(W: tl.View[ggml_formats.Q3_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
              Y: tl.View[tl.f32, (1,)], *, LANES: tl.constexpr = 32):
    tl.store(Y[0], compute_lanes(W, X, LANES=LANES))
