from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as tl


@triton.jit
def lane_scaled(W, X: tl.View[ggml.Q8_K, (K,)], CHUNK: tl.constexpr):
    """Keep exact signed products and their group scales in the logical lanes."""
    tl.static_assert(CHUNK == 64 or CHUNK == 128 or CHUNK == 256)
    lane = tl.arange(0, CHUNK, dtype=tl.u32, axis="k")
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        x = tl.load(X[block])
        total = tl.i32(0)
        for part in tl.static_range(256 // CHUNK):
            coordinate = lane + tl.u32(part * CHUNK)
            code = tl.u8(w.ql[coordinate]) | (tl.u8(w.qh[coordinate]) << tl.u8(4))
            weight = tl.i8(code) - tl.i8(32)
            product = tl.i16(weight) * tl.i16(x.q[coordinate])
            scaled = tl.i32(product) * tl.i32(w.scales[coordinate // tl.u32(16)])
            total += tl.sum(scaled, 0)
        result += (tl.f32(w.d) * tl.f32(x.ds)) * tl.f32(total)
    return result


@triton.jit(model="weft")
def dot(W: tl.View[ggml.Q6_K, (K,)], X: tl.View[ggml.Q8_K, (K,)],
        Y: tl.View[tl.f32, (1,)], *, CHUNK: tl.constexpr):
    tl.store(Y[0], lane_scaled(W, X, CHUNK))


@triton.jit
def plane_partials(w, x, LANES: tl.constexpr, PLANES: tl.constexpr):
    plane = tl.arange(0, PLANES, dtype=tl.u32, axis="plane")
    lane = tl.arange(0, LANES, dtype=tl.u32, axis="lane")
    accumulated = tl.i32(lane) * tl.i32(0)
    for group in tl.static_range(256 // (LANES * PLANES)):
        coordinate = tl.u32(group * LANES * PLANES) + plane * tl.u32(LANES) + lane
        code = tl.u8(w.ql[coordinate]) | (tl.u8(w.qh[coordinate]) << tl.u8(4))
        weight = tl.i8(code) - tl.i8(32)
        product = tl.i16(weight) * tl.i16(x.q[coordinate])
        scaled = tl.i32(product) * tl.i32(w.scales[coordinate // tl.u32(16)])
        accumulated += tl.reduce(scaled, axis="plane")
    return accumulated


@triton.jit
def deferred_planes(W: tl.View[ggml.Q6_K, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                    LANES: tl.constexpr, PLANES: tl.constexpr):
    """Retain float lanes across the blocks of one encoded row."""
    tl.static_assert(LANES == 32 or LANES == 64 or LANES == 128)
    tl.static_assert(PLANES >= 2 and 256 % (LANES * PLANES) == 0)
    lane = tl.arange(0, LANES, dtype=tl.u32, axis="lane")
    result = tl.f32(lane) * tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        x = tl.load(X[block])
        accumulated = plane_partials(w, x, LANES, PLANES)
        result += (tl.f32(w.d) * tl.f32(x.ds)) * tl.f32(accumulated)
    return tl.reduce(result, axis="lane")


@triton.jit(model="weft")
def dot_deferred_planes(W: tl.View[ggml.Q6_K, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                        Y: tl.View[tl.f32, (1,)], *, LANES: tl.constexpr,
                        PLANES: tl.constexpr, K: tl.constexpr):
    """Specialize a projection width while sharing the dynamic-row program."""
    tl.static_assert(K % 256 == 0)
    tl.store(Y[0], deferred_planes(W, X, LANES, PLANES))


@triton.jit(model="weft")
def dot_deferred(W: tl.View[ggml.Q6_K, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                 Y: tl.View[tl.f32, (1,)], *, LANES: tl.constexpr,
                 PLANES: tl.constexpr):
    """Dynamic K entry for framework workers consuming ordinary GGML rows."""
    tl.store(Y[0], deferred_planes(W, X, LANES, PLANES))


@triton.jit
def block_scaled(W: tl.View[ggml.Q6_K, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                 LANES: tl.constexpr, PLANES: tl.constexpr):
    """Finish each integer block before applying its floating-point scale."""
    tl.static_assert(LANES == 32 or LANES == 64 or LANES == 128)
    tl.static_assert(PLANES >= 2 and 256 % (LANES * PLANES) == 0)
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        x = tl.load(X[block])
        integer = tl.reduce(plane_partials(w, x, LANES, PLANES), axis="lane")
        result += (tl.f32(w.d) * tl.f32(x.ds)) * tl.f32(integer)
    return result


@triton.jit(model="weft")
def dot_block_scaled(W: tl.View[ggml.Q6_K, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                     Y: tl.View[tl.f32, (1,)], *, LANES: tl.constexpr,
                     PLANES: tl.constexpr):
    tl.store(Y[0], block_scaled(W, X, LANES, PLANES))


@triton.jit
def stream_scale_groups(W: tl.View[ggml.Q6_K, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                        GROUPS: tl.constexpr = 1):
    tl.static_assert(GROUPS == 1 or GROUPS == 2 or GROUPS == 4 or GROUPS == 8)
    lane = tl.arange(0, 16, dtype=tl.u8, axis="k")
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        total = tl.i32(0)
        for window in tl.static_range(16 // GROUPS):
            for offset in tl.static_range(GROUPS):
                group = window * GROUPS + offset
                coordinate = tl.u8(group) * tl.u8(16) + lane
                code = tl.i8(w.ql[coordinate]) | (tl.i8(w.qh[coordinate]) << tl.u8(4))
                partial = tl.reduce_dot(x.q[coordinate], code - tl.i8(32), over="k", acc_dtype=tl.i32)
                total += partial * tl.i32(w.scales[group])
        result += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(total)
    return result


@triton.jit(model="weft")
def dot_stream_scale_groups(W: tl.View[ggml.Q6_K, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                            Y: tl.View[tl.f32, (1,)], *, GROUPS: tl.constexpr = 1):
    tl.store(Y[0], stream_scale_groups(W, X, GROUPS))
