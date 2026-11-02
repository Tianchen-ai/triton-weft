from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit
def compute_tq2(W: tl.View[ggml_formats.TQ2_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                LANES: tl.constexpr, DEFERRED: tl.constexpr = False):
    """Consume bounded code windows and the activation's stored group sums."""
    tl.static_assert(LANES == 16 or LANES == 32 or LANES == 64 or LANES == 128)
    lane = tl.arange(0, LANES, dtype=tl.u32, axis="k")
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        integer = tl.i32(0)
        if DEFERRED:
            partial = tl.i16(lane) * tl.i16(0)
        for part in tl.static_range(256 // LANES):
            coordinate = lane + tl.u32(part * LANES)
            if DEFERRED:
                # Even all sixteen windows at the largest two-bit code fit
                # i16: 16 * 128 * 3 = 6144. Widen before the final reduction.
                partial += tl.i16(x.q[coordinate]) * tl.i16(tl.i8(w.q[coordinate]))
            else:
                integer += tl.reduce_dot(x.q[coordinate], tl.i8(w.q[coordinate]), over="k", acc_dtype=tl.i32)
        if DEFERRED:
            integer = tl.reduce(tl.i32(partial), axis="k")
        integer -= tl.reduce(tl.i32(x.bsum), axis="k")
        result += tl.f32(integer) * (tl.f32(w.d) * tl.f32(x.ds))
    return result


@triton.jit(model="weft")
def tq2_0(W: tl.View[ggml_formats.TQ2_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          Y: tl.View[tl.f32, (1,)], *, LANES: tl.constexpr, DEFERRED: tl.constexpr = False):
    tl.store(Y[0], compute_tq2(W, X, LANES, DEFERRED))


@triton.jit
def compute_tq1(W: tl.View[ggml_formats.TQ1_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                powers: tl.View[tl.u32, (5,)]):
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        lane32 = tl.arange(0, 32, dtype=tl.u32, axis="k")
        lane16 = tl.arange(0, 16, dtype=tl.u32, axis="k")
        lane4 = tl.arange(0, 4, dtype=tl.u32, axis="k")
        sum32 = tl.i16(lane32 * tl.u32(0))
        sum16 = tl.i16(lane16 * tl.u32(0))
        sum4 = tl.i16(lane4 * tl.u32(0))
        # Each decoded digit is 0..2; five Q8 products fit exactly in i16.
        # Keep independent lane partials until each stored window is consumed.
        for digit in tl.static_range(5):
            power = tl.u8(tl.load(powers[digit]))
            wrapped32 = tl.u8(w.q[lane32]) * power
            code32 = tl.i8((tl.u16(wrapped32) * tl.u16(3)) >> tl.u16(8))
            sum32 += tl.i16(x.q[tl.u32(digit * 32) + lane32]) * tl.i16(code32)
        for digit in tl.static_range(5):
            power = tl.u8(tl.load(powers[digit]))
            wrapped16 = tl.u8(w.q[tl.u32(32) + lane16]) * power
            code16 = tl.i8((tl.u16(wrapped16) * tl.u16(3)) >> tl.u16(8))
            sum16 += tl.i16(x.q[tl.u32(160 + digit * 16) + lane16]) * tl.i16(code16)
        for digit in tl.static_range(4):
            power = tl.u8(tl.load(powers[digit]))
            wrapped4 = tl.u8(w.qh[lane4]) * power
            code4 = tl.i8((tl.u16(wrapped4) * tl.u16(3)) >> tl.u16(8))
            sum4 += tl.i16(x.q[tl.u32(240 + digit * 4) + lane4]) * tl.i16(code4)
        integer = tl.sum(tl.i32(sum32)) + tl.sum(tl.i32(sum16)) + tl.sum(tl.i32(sum4))
        integer -= tl.reduce(tl.i32(x.bsum), axis="k")
        result += tl.f32(integer) * (tl.f32(w.d) * tl.f32(x.ds))
    return result


@triton.jit(model="weft")
def tq1_0(W: tl.View[ggml_formats.TQ1_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          powers: tl.View[tl.u32, (5,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_tq1(W, X, powers))
