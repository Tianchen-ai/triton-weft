from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from .q4_k_q8_k import compute as compute_affine

G16 = tl.constexpr(16)
G8 = tl.constexpr(8)
LANES = tl.constexpr(64)

@triton.jit
def scaled_quarters(q, x, scales, TERMS: tl.constexpr):
    """Keep scaled four-term partials live until the end of the row."""
    # These callers declare bounded unsigned weights and signed bytes. A pair
    # fits I16; multiplication by the signed group scale accumulates in I32.
    pairs = tl.mac_pairs(q, x, into=tl.i16)
    coordinate = tl.arange(0, 128, dtype=tl.u32, axis="k")
    pair_scales = tl.i16(scales[coordinate // tl.u32(TERMS // 2)])
    quarters = tl.mac_pairs(pairs, pair_scales, into=tl.i32)
    return tl.reshape(quarters, (LANES,), axes=("LANES",), order=("k",))


@triton.jit
def compute_q2(W, X: tl.View[ggml_formats.Q8_K, (K,)]):
    result = tl.state(tl.f32, [G16], init=0)
    minimum_result = tl.state(tl.f32, [G16], init=0)
    with tl.level.blocks(K, extent=256) as kb:
        w, x = tl.load(W[kb]), tl.load(X[kb])
        metadata = tl.f32(tl.join(w.d, w.dmin)) * tl.f32(x.ds)
        integer = tl.sum(tl.reshape(scaled_quarters(w.q, x.q, tl.u4(w.scales), 16),
                         (4, G16), axes=("segment", "G16"), order=("LANES",)), 0)
        minimum = tl.reshape(tl.i32(x.bsum) * tl.i32(tl.u8(w.scales) >> 4),
                             (16,), axes=("G16",), order=("k",))
        result += metadata[0] * tl.f32(integer)
        minimum_result += metadata[1] * tl.f32(minimum)
    return tl.sum(result, 0) - tl.sum(minimum_result, 0)


@triton.jit
def compute_q3(W, X: tl.View[ggml_formats.Q8_K, (K,)]):
    result = tl.state(tl.f32, [G16], init=0)
    with tl.level.blocks(K, extent=256) as kb:
        w, x = tl.load(W[kb]), tl.load(X[kb])
        q = tl.u4(tl.u8(w.q) | (tl.u8(w.hmask) << 2))
        decoded = tl.i8((tl.u8(w.scale_low) | (tl.u8(w.scale_high) << 4)) - tl.u8(32))
        integer = tl.sum(tl.reshape(scaled_quarters(q, x.q, decoded, 16),
                         (4, G16), axes=("segment", "G16"), order=("LANES",)), 0)
        centered = tl.reshape(tl.i32(x.bsum) * tl.i32(decoded) * 4,
                              (16,), axes=("G16",), order=("k",))
        result += (tl.f32(w.d) * tl.f32(x.ds)) * tl.f32(integer - centered)
    return tl.sum(result, 0)


@triton.jit
def compute_q5(W, X: tl.View[ggml_formats.Q8_K, (K,)], LANES: tl.constexpr = 8):
    return compute_affine(W, X, LANES, HIGH=True)


@triton.jit(model="weft")
def q2_k(W: tl.View[ggml_formats.Q2_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_q2(W, X))


@triton.jit(model="weft")
def q3_k(W: tl.View[ggml_formats.Q3_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_q3(W, X))


@triton.jit(model="weft")
def q5_k(W: tl.View[ggml_formats.Q5_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)],
         LANES: tl.constexpr = 8):
    tl.store(Y[0], compute_q5(W, X, LANES))
