from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl

PAIRS = tl.constexpr(8)
SHORT_PAIRS = tl.constexpr(8)


@triton.jit
def compute_q1_0(W, X: tl.View[ggml_formats.Q8_0, (K,)]):
    first = tl.state(tl.f32, [PAIRS], init=0)
    second = tl.state(tl.f32, [PAIRS], init=0)
    lane = tl.arange(0, 32, dtype=tl.u32, axis="k")
    with tl.level.blocks(K, extent=128) as block:
        w = tl.load(W[block])
        for part in tl.static_range(4):
            x = tl.load(tl.subview(X[block], offsets=(part * 32,), extents=(32,)))
            positive = w.q[lane + tl.u32(part * 32)]
            activation = tl.where(positive != 0, x.q, -x.q)
            products = tl.mac_groups(tl.u4(positive) * tl.u4(0) + tl.u4(1), activation, n=4, into=tl.i32)
            scaled = (tl.f32(w.d) * tl.f32(x.d)) * tl.f32(products)
            contribution = tl.reshape(scaled, (PAIRS,), axes=("PAIRS",), order=("k",))
            if part % 2 == 0:
                first += contribution
            else:
                second += contribution
    return tl.sum(first + second, 0)


@triton.jit
def compute_nvfp4(W, X: tl.View[ggml_formats.Q8_0, (K,)], codebook, scale):
    table = tl.u8(tl.load(codebook)) + tl.u8(128)
    first = tl.state(tl.f32, [SHORT_PAIRS], init=0)
    second = tl.state(tl.f32, [SHORT_PAIRS], init=0)
    lane = tl.arange(0, 32, dtype=tl.u32, axis="k")
    group = tl.arange(0, SHORT_PAIRS, dtype=tl.u32, axis="k") // tl.u32(4)
    with tl.level.blocks(K, extent=64) as block:
        w = tl.load(W[block])
        weight = tl.lookup(table, tl.u8(w.q), bounds="in_bounds")
        for part in tl.static_range(2):
            x = tl.load(tl.subview(X[block], offsets=(part * 32,), extents=(32,)))
            products = tl.mac_groups(weight[lane + tl.u32(part * 32)], x.q, n=4, into=tl.i32)
            bias = tl.u8(x.q) * tl.u8(0) + tl.u8(128)
            products -= tl.mac_groups(bias, x.q, n=4, into=tl.i32)
            first_scale = tl.load(scale[tl.index(w.d[part * 2])])
            second_scale = tl.load(scale[tl.index(w.d[part * 2 + 1])])
            exponent = tl.where(group == tl.u32(0), first_scale, second_scale)
            scaled = (tl.f32(x.d) * exponent) * tl.f32(products)
            contribution = tl.reshape(scaled, (SHORT_PAIRS,), axes=("SHORT_PAIRS",), order=("k",))
            if part == 0:
                first += contribution
            else:
                second += contribution
    return tl.sum(first + second, 0)


@triton.jit
def compute_q1_factored(W, X: tl.View[ggml_formats.Q8_0, (K,)]):
    result = tl.state(tl.f32, [PAIRS], init=0)
    lane = tl.arange(0, 32, dtype=tl.u32, axis="k")
    with tl.level.blocks(K, extent=128) as block:
        w = tl.load(W[block])
        local = tl.state(tl.f32, [PAIRS], init=0)
        for part in tl.static_range(4):
            x = tl.load(tl.subview(X[block], offsets=(part * 32,), extents=(32,)))
            positive = w.q[lane + tl.u32(part * 32)]
            signed = tl.where(positive != 0, x.q, -x.q)
            products = tl.mac_groups(tl.u4(positive) * tl.u4(0) + tl.u4(1), signed, n=4, into=tl.i32)
            contribution = tl.reshape(tl.f32(products), (PAIRS,), axes=("PAIRS",), order=("k",))
            local += tl.f32(x.d) * contribution
        result += tl.f32(w.d) * local
    return tl.sum(result, 0)


@triton.jit(model="weft")
def q1_0_factored(W: tl.View[ggml_formats.Q1_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)],
                  Y: tl.View[tl.f32, (1,)], *, K: tl.constexpr):
    tl.store(Y[0], compute_q1_factored(W, X))


@triton.jit(model="weft")
def q1_0(W: tl.View[ggml_formats.Q1_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_q1_0(W, X))


@triton.jit(model="weft")
def q1_0_specialized(W: tl.View[ggml_formats.Q1_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)],
                     Y: tl.View[tl.f32, (1,)], *, K: tl.constexpr):
    tl.store(Y[0], compute_q1_0(W, X))


@triton.jit(model="weft")
def nvfp4(W: tl.View[ggml_formats.NVFP4, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)],
           codebook: tl.View[tl.i8, (16,)], scale: tl.View[tl.f32, (256,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_nvfp4(W, X, codebook, scale))


@triton.jit(model="weft")
def nvfp4_specialized(W: tl.View[ggml_formats.NVFP4, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)],
                      codebook: tl.View[tl.i8, (16,)], scale: tl.View[tl.f32, (256,)],
                      Y: tl.View[tl.f32, (1,)], *, K: tl.constexpr):
    tl.store(Y[0], compute_nvfp4(W, X, codebook, scale))
