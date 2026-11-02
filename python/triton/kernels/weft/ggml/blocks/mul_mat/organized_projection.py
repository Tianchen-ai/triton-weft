"""K-block projections composed from stored fields and shared traversal helpers."""
from __future__ import annotations

import triton
import triton.language as tl
from triton.integrations import ggml_formats
from ..quantize import q8_k


@triton.jit
def q4_fields(weight):
    return weight.q, tl.i32(weight.sc)


@triton.jit
def q4_minima(weight):
    return tl.i32(weight.m), tl.f32(weight.dmin)


@triton.jit
def q6_fields(weight):
    codes = (tl.i8(weight.ql) | (tl.i8(weight.qh) << tl.u8(4))) - tl.i8(32)
    return codes, tl.i32(weight.scales)


def make_projection(encoding, fields, partials_per_scale, correction=None):
    """Bind format knowledge once; every instance has the same typed interface."""
    group_size = tl.constexpr(partials_per_scale)

    @triton.jit
    def prepare(tile, SHARED: tl.constexpr):
        weight = tl.load(tile)
        codes, scales = fields(weight)
        if correction is not None:
            minima, minimum_scale = correction(weight)
        scale = tl.f32(weight.d)
        if SHARED:
            codes, scales, scale = tl.stage(codes), tl.stage(scales), tl.stage(scale)
        if correction is not None:
            if SHARED:
                minima, minimum_scale = tl.stage(minima), tl.stage(minimum_scale)
            return codes, scales, scale, minima, minimum_scale
        else:
            return codes, scales, scale

    @triton.jit
    def calculate(prepared, activation, GROUPED: tl.constexpr):
        codes, scales, scale = prepared[0], prepared[1], prepared[2]
        coordinate = tl.arange(0, 64, dtype=tl.u32, axis="K")
        if GROUPED:
            partial = tl.widen(tl.mac_groups(activation.q, codes, n=4, into=tl.i16), tl.i32)
        else:
            partial = tl.mac_groups(activation.q, codes, n=4, into=tl.i32)
        integer = tl.reduce(partial * scales[:, coordinate // group_size], axis="K")
        value = activation.ds * scale * tl.f32(integer)
        if correction is not None:
            minima, minimum_scale = prepared[3], prepared[4]
            offset = tl.dot(tl.sum_pairs(activation.bsum), minima,
                            over="K", acc_dtype=tl.i32, engine="vector")
            value -= activation.ds * minimum_scale * tl.f32(offset)
        return value

    @triton.jit
    def update_outputs(W, Xq, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       MR: tl.constexpr, NR: tl.constexpr,
                       SHARED: tl.constexpr, GROUPED: tl.constexpr):
        for m in range(M):
            for n in range(N):
                tl.store(Y[m, n], tl.f32(0))
        with tl.level.cols(N, group=NR) as columns:
            with tl.level.blocks(K, extent=256) as block:
                if SHARED:
                    prepared = prepare(W[columns, block], SHARED)
                with tl.level.rows(M, group=MR) as rows:
                    if not SHARED:
                        prepared = prepare(W[columns, block], SHARED)
                    value = calculate(prepared, tl.load(Xq[rows, block]), GROUPED)
                    tl.store(Y[rows, columns], tl.load(Y[rows, columns]) + value)

    @triton.jit
    def retain_outputs(W, Xq, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       MR: tl.constexpr, NR: tl.constexpr, MC: tl.constexpr,
                       SHARED: tl.constexpr, GROUPED: tl.constexpr):
        with tl.level.rows(M, group=MC) as cohort:
            with tl.level.cols(N, group=NR) as columns:
                accumulator = tl.state(tl.f32, [MC, NR], init=0)
                with tl.level.blocks(K, extent=256) as block:
                    if SHARED:
                        prepared = prepare(W[columns, block], SHARED)
                    with tl.level.rows(cohort, group=MR) as rows:
                        if not SHARED:
                            prepared = prepare(W[columns, block], SHARED)
                        value = calculate(prepared, tl.load(Xq[rows, block]), GROUPED)
                        if MC == MR:
                            accumulator += value
                        else:
                            accumulator[rows] = accumulator[rows] + value
                tl.store(Y[cohort, columns], accumulator)

    @triton.jit(model="weft", alias_groups={"W": "weights", "X": "input", "Xq": "workspace", "Y": "output"})
    def projection(W: tl.View[encoding, (N, K)], X: tl.View[tl.f32, (M, K)],
                   Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
                   MR: tl.constexpr, NR: tl.constexpr, SHARED: tl.constexpr,
                   GROUPED: tl.constexpr, RETAIN_OUTPUT: tl.constexpr = False,
                   MC: tl.constexpr = 16):
        q8_k.quantize_matrix(X, Xq)
        if RETAIN_OUTPUT:
            retain_outputs(W, Xq, Y, M, N, K, MR, NR, MC, SHARED, GROUPED)
        else:
            update_outputs(W, Xq, Y, M, N, K, MR, NR, SHARED, GROUPED)

    return projection


q4_projection = make_projection(ggml_formats.Q4_K, q4_fields, 8, q4_minima)
q6_projection = make_projection(ggml_formats.Q6_K, q6_fields, 4)
