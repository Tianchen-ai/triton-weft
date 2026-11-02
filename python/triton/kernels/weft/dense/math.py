"""Ordinary bounded numerical programs for range-sensitive physical lowering."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", strided=("X", "Y"))
def exp_nonpositive(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tl.store(Y[row, part], tl.exp(-tl.abs(tl.load(X[row, part]))))


@triton.jit(model="weft", strided=("X", "Y"))
def exp_clamped(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.maximum(tl.minimum(tl.load(X[row, part]), 87.0), -87.0)
            tl.store(Y[row, part], tl.exp(value))


@triton.jit(model="weft", strided=("X", "Y"))
def log_clamped(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.maximum(tl.minimum(tl.load(X[row, part]), 1.2676506002282294e30), 7.888609052210118e-31)
            tl.store(Y[row, part], tl.log(value))


@triton.jit(model="weft", strided=("X", "Y"))
def log_exp_normalizer(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tl.store(Y[row, part], tl.log(1.0 + tl.exp(-tl.abs(tl.load(X[row, part])))))


@triton.jit(model="weft", strided=("X", "Y"))
def selected_exp(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.load(X[row, part])
            upper = tl.select(value > 5.0, 5.0, value)
            bounded = tl.select(upper < -5.0, -5.0, upper)
            tl.store(Y[row, part], tl.exp(bounded))


@triton.jit(model="weft", strided=("X", "Y"))
def masked_exp(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.load(X[row, part])
            in_range = (value >= -5.0) & (value <= 5.0)
            tl.store(Y[row, part], tl.exp(tl.select(in_range, value, 0.0)))


@triton.jit(model="weft", strided=("X", "Y"))
def complemented_exp(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.load(X[row, part])
            outside = ((value >= -5.0) & (value <= 5.0)) ^ tl.i1(True)
            tl.store(Y[row, part], tl.exp(tl.select(outside, 0.0, value)))
