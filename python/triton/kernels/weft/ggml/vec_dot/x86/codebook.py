from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl

PAIRS = tl.constexpr(8)
QUARTERS = tl.constexpr(64)
PARTIALS = tl.constexpr(16)


@triton.jit
def pairs32(q, x, table):
    decoded = tl.lookup(table, tl.u8(q), bounds="in_bounds")
    partial = tl.mac_groups(decoded, x, n=4, into=tl.i32)
    return tl.reshape(partial, (PAIRS,), axes=("PAIRS",), order=("k",))


@triton.jit
def compute_iq4_nl(W, X: tl.View[ggml_formats.Q8_0, (K,)], codebook):
    table = tl.load(codebook)
    first = tl.state(tl.f32, [PAIRS], init=0)
    second = tl.state(tl.f32, [PAIRS], init=0)
    with tl.level.blocks(K, extent=64) as kb:
        for part in tl.static_range(2):
            if tl.extent(W[kb], axis=0) >= (part + 1) * 32:
                w = tl.load(tl.subview(W[kb], offsets=(part * 32,), extents=(32,)))
                x = tl.load(tl.subview(X[kb], offsets=(part * 32,), extents=(32,)))
                metadata = tl.f32(tl.join(w.d, x.d))
                scaled = (metadata[0] * metadata[1]) * tl.f32(pairs32(w.q, x.q, table))
                if part == 0:
                    first += scaled
                else:
                    second += scaled
    return tl.sum(first + second, 0)


@triton.jit
def compute_mxfp4(W, X: tl.View[ggml_formats.Q8_0, (K,)], codebook, scale):
    table = tl.load(codebook)
    first = tl.state(tl.f32, [PAIRS], init=0)
    second = tl.state(tl.f32, [PAIRS], init=0)
    with tl.level.blocks(K, extent=64) as kb:
        for part in tl.static_range(2):
            if tl.extent(W[kb], axis=0) >= (part + 1) * 32:
                w = tl.load(tl.subview(W[kb], offsets=(part * 32,), extents=(32,)))
                x = tl.load(tl.subview(X[kb], offsets=(part * 32,), extents=(32,)))
                exponent = tl.load(scale[tl.index(w.e)])
                value = (tl.f32(x.d) * exponent) * tl.f32(pairs32(w.q, x.q, table))
                if part == 0:
                    first += value
                else:
                    second += value
    return tl.sum(first + second, 0)


@triton.jit
def compute_iq4_xs(W, X: tl.View[ggml_formats.Q8_K, (K,)], codebook, PARTIALS: tl.constexpr = 16):
    tl.static_assert(PARTIALS == 8 or PARTIALS == 16)
    table = tl.u8(tl.load(codebook)) + tl.u8(128)
    first = tl.state(tl.f32, [PARTIALS], init=0)
    second = tl.state(tl.f32, [PARTIALS], init=0)
    with tl.level.blocks(K, extent=512) as kb:
        for part in tl.static_range(2):
            if tl.extent(W[kb], axis=0) >= (part + 1) * 256:
                w = tl.load(tl.subview(W[kb], offsets=(part * 256,), extents=(256,)))
                x = tl.load(tl.subview(X[kb], offsets=(part * 256,), extents=(256,)))
                contribution = iq4_xs_block(w, x, table, PARTIALS)
                if part == 0:
                    first += contribution
                else:
                    second += contribution
    return tl.sum(first + second, 0)


@triton.jit
def iq4_xs_block(w, x, table, PARTIALS: tl.constexpr):
    coordinate = tl.arange(0, 64, dtype=tl.u32, axis="k")
    group = tl.arange(0, 8, dtype=tl.u16, axis="group")
    correction_group = tl.arange(0, 16, dtype=tl.u16, axis="sum_group")
    decoded = tl.lookup(table, tl.u8(w.q), bounds="in_bounds")
    quarters = tl.mac_groups(decoded, x.q, n=4, into=tl.i32)
    low = tl.u8(w.scales_l[group // 2])
    scale_low = (low >> tl.u8((group % 2) * 4)) & tl.u8(15)
    scale_high = tl.u8((tl.u16(w.scales_h) >> tl.u16(group * 2)) & tl.u16(3))
    scales = tl.i32(tl.i8(scale_low | (scale_high << 4)) - tl.i8(32))
    scaled = tl.sum(tl.reshape(quarters * scales[coordinate // 8], (64 // PARTIALS, PARTIALS),
                                   axes=("segment", "PARTIALS"), order=("k",)), 0)
    correction = tl.i32(x.bsum[correction_group]) * scales[correction_group // 2] * 128
    correction = tl.sum(tl.reshape(correction, (16 // PARTIALS, PARTIALS),
                                      axes=("segment", "PARTIALS"), order=("sum_group",)), 0)
    return (tl.f32(w.d) * tl.f32(x.ds)) * tl.f32(scaled - correction)


@triton.jit(model="weft")
def iq4_nl(W: tl.View[ggml_formats.IQ4_NL, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)],
           codebook: tl.View[tl.i8, (16,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_iq4_nl(W, X, codebook))


@triton.jit(model="weft")
def iq4_nl_specialized(W: tl.View[ggml_formats.IQ4_NL, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)],
                       codebook: tl.View[tl.i8, (16,)], Y: tl.View[tl.f32, (1,)], *, K: tl.constexpr):
    tl.store(Y[0], compute_iq4_nl(W, X, codebook))


@triton.jit(model="weft")
def mxfp4(W: tl.View[ggml_formats.MXFP4, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)],
          codebook: tl.View[tl.i8, (16,)], scale: tl.View[tl.f32, (256,)],
          Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_mxfp4(W, X, codebook, scale))


@triton.jit(model="weft")
def iq4_xs(W: tl.View[ggml_formats.IQ4_XS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
           codebook: tl.View[tl.i8, (16,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_iq4_xs(W, X, codebook))


@triton.jit(model="weft")
def iq4_xs_specialized(W: tl.View[ggml_formats.IQ4_XS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                      codebook: tl.View[tl.i8, (16,)], Y: tl.View[tl.f32, (1,)], *, K: tl.constexpr,
                      PARTIALS: tl.constexpr = 16):
    tl.store(Y[0], compute_iq4_xs(W, X, codebook, PARTIALS))
