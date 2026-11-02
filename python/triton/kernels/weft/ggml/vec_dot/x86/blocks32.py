from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl

@triton.jit
def compute(W, X, K, CENTER: tl.constexpr, HIGH: tl.constexpr, AFFINE: tl.constexpr,
            GROUP: tl.constexpr = 2):
    LANES: tl.constexpr = 32 // GROUP
    partial = tl.state(tl.f32, [LANES], init=0)
    correction = tl.f32(0)
    with tl.level.blocks(K, extent=32) as kb:
        w, x = tl.load(W[kb]), tl.load(X[kb])
        if HIGH:
            q = tl.u8(w.q) | (tl.u8(w.qh) << 4)
        else:
            q = w.q
        if CENTER:
            centered = tl.i8(q) - tl.i8(CENTER)
        else:
            centered = q
        pairs = tl.mac_groups(centered, x.q, n=GROUP, into=tl.i32)
        values = tl.reshape(pairs, (LANES,), axes=("LANES",), order=("k",))
        if AFFINE:
            weight_metadata = tl.f32(tl.join(w.d, w.m))
            activation_metadata = tl.f32(tl.join(x.d, x.s))
            metadata = weight_metadata * activation_metadata
            partial += metadata[0] * tl.f32(values)
            correction += metadata[1]
        else:
            partial += (tl.f32(w.d) * tl.f32(x.d)) * tl.f32(values)
    return tl.sum(partial, 0) + correction


@triton.jit
def compute_centered_pairs(W, X, K, CENTER: tl.constexpr, HIGH: tl.constexpr):
    LANES: tl.constexpr = 8
    first = tl.state(tl.f32, [LANES], init=0)
    second = tl.state(tl.f32, [LANES], init=0)
    with tl.level.blocks(K, extent=64) as kb:
        for part in tl.static_range(2):
            if part * 32 < tl.extent(W[kb], axis=0):
                w = tl.load(tl.subview(W[kb], offsets=(part * 32,), extents=(32,)))
                x = tl.load(tl.subview(X[kb], offsets=(part * 32,), extents=(32,)))
                if HIGH:
                    q = tl.u8(w.q) | (tl.u8(w.qh) << 4)
                else:
                    q = tl.u8(w.q)
                centered = tl.i8(q) - tl.i8(CENTER)
                products = tl.mac_groups(centered, x.q, n=4, into=tl.i32)
                scaled = (tl.f32(w.d) * tl.f32(x.d)) * tl.f32(products)
                value = tl.reshape(scaled, (LANES,), axes=("LANES",), order=("k",))
                if part == 0:
                    first += value
                else:
                    second += value
    return tl.sum(first + second, 0)


@triton.jit(model="weft")
def q4_0(W: tl.View[ggml_formats.Q4_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute(W, X, K, 8, False, False, 4))


@triton.jit(model="weft")
def q4_1(W: tl.View[ggml_formats.Q4_1, (K,)], X: tl.View[ggml_formats.Q8_1, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute(W, X, K, 0, False, True, 4))


@triton.jit(model="weft")
def q5_0(W: tl.View[ggml_formats.Q5_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_centered_pairs(W, X, K, 16, True))


@triton.jit(model="weft")
def q5_1(W: tl.View[ggml_formats.Q5_1, (K,)], X: tl.View[ggml_formats.Q8_1, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute(W, X, K, 0, True, True, 4))


@triton.jit(model="weft")
def q8_0(W: tl.View[ggml_formats.Q8_0, (K,)], X: tl.View[ggml_formats.Q8_0, (K,)],
         Y: tl.View[tl.f32, (1,)], GROUP: tl.constexpr = 4):
    tl.static_assert(GROUP == 2 or GROUP == 4)
    tl.store(Y[0], compute(W, X, K, 0, False, False, GROUP))
