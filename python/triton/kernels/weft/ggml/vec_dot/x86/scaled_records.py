from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from ...mul_mat.x86.scaled_records import (
    decode_iq1_m, decode_iq1, iq2_components,
)


@triton.jit
def compute(W, X: tl.View[ggml_formats.Q8_K, (K,)], grid, auxiliary,
            DECODE: tl.constexpr):
    PARTIALS: tl.constexpr = 16
    partial = tl.state(tl.f32, [PARTIALS], init=0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        weights, d = DECODE(w, grid, auxiliary)
        activation = tl.reshape(x.q, (256,), axes=("k",), order=("K",))
        pairs = tl.mac_pairs(weights, activation, into=tl.i32)
        groups = tl.reshape(pairs, (8, PARTIALS), axes=("group", "PARTIALS"), order=("k",))
        integer = tl.sum(groups, 0)
        partial += tl.f32(integer) * (d * tl.f32(x.ds))
    return tl.sum(partial, 0)


@triton.jit
def compute_grouped(W, X: tl.View[ggml_formats.Q8_K, (K,)], grid,
                    VARIANT: tl.constexpr):
    PARTIALS: tl.constexpr = 16
    partial = tl.state(tl.f32, [PARTIALS], init=0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        weight, sign, scale, d = iq2_components(w, grid, None, VARIANT, True)
        activation = tl.reshape(x.q, (32, 8), axes=("entry", "payload"), order=("K",))
        # The format's positive grid entries fit U6. Q8_K quantization produces
        # [-127,127], so transferring the sign to activation is exact.
        activation = tl.where(sign < 0, -activation, activation)
        pairs = tl.mac_pairs(tl.u6(weight), activation, into=tl.i16)
        pair_lane = tl.arange(0, 4, dtype=tl.u16, axis="payload")
        scaled = tl.mac_pairs(pairs, scale + tl.i16(pair_lane * 0), into=tl.i32)
        groups = tl.reshape(scaled, (4, PARTIALS), axes=("group", "PARTIALS"), order=("entry", "payload"))
        partial += tl.f32(tl.sum(groups, 0)) * (d * tl.f32(x.ds))
    return tl.sum(partial, 0)


@triton.jit(model="weft")
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          grid: tl.View[ggml_formats.I8X8, (2048, 8)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_iq1_s(W, X, grid))


@triton.jit
def compute_iq1_s(W, X: tl.View[ggml_formats.Q8_K, (K,)], grid):
    PARTIALS: tl.constexpr = 16
    partial = tl.state(tl.f32, [PARTIALS], init=0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        weight, scale, delta, d = decode_iq1(w, grid, None, False, POSITIVE=True)
        activation = tl.reshape(x.q, (32, 8), axes=("entry", "payload"), order=("K",))
        products = tl.mac_groups(weight, activation, n=4, into=tl.i32)
        scaled = products * tl.i32(scale) * tl.i32(8)
        groups = tl.reshape(scaled, (4, PARTIALS), axes=("segment", "PARTIALS"), order=("entry", "payload"))
        correction_scale = tl.reshape((tl.i32(8) - tl.i32(delta)) * tl.i32(scale),
                                      (PARTIALS, 2), axes=("PARTIALS", "pair"), order=("entry",))
        sums = tl.reshape(tl.i32(x.bsum), (PARTIALS,), axes=("PARTIALS",), order=("K",))
        total = tl.sum(groups, 0) - correction_scale[:, 0] * sums
        partial += tl.f32(total) * (d * tl.f32(x.ds))
    return tl.sum(partial, 0) * tl.f32(0.125)


@triton.jit(model="weft")
def iq1_m(W: tl.View[ggml_formats.IQ1_M, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          grid: tl.View[ggml_formats.I8X8, (2048, 8)], f16_bits: tl.View[tl.f32, (65536,)],
          Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute(W, X, grid, f16_bits, decode_iq1_m))


@triton.jit(model="weft")
def iq2_xs(W: tl.View[ggml_formats.IQ2_XS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
           grid: tl.View[ggml_formats.I8X8, (512, 8)],
           Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_grouped(W, X, grid, True))


@triton.jit(model="weft")
def iq2_xs_specialized(W: tl.View[ggml_formats.IQ2_XS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                       grid: tl.View[ggml_formats.I8X8, (512, 8)],
                       Y: tl.View[tl.f32, (1,)], *, K: tl.constexpr):
    tl.store(Y[0], compute_grouped(W, X, grid, True))


@triton.jit(model="weft")
def iq2_xxs(W: tl.View[ggml_formats.IQ2_XXS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
            grid: tl.View[ggml_formats.I8X8, (256, 8)],
            Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_grouped(W, X, grid, False))


@triton.jit(model="weft")
def iq2_xxs_specialized(W: tl.View[ggml_formats.IQ2_XXS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                       grid: tl.View[ggml_formats.I8X8, (256, 8)],
                       Y: tl.View[tl.f32, (1,)], *, K: tl.constexpr):
    tl.store(Y[0], compute_grouped(W, X, grid, False))
