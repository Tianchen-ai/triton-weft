from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def statistics(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (3,)]):
    value = tl.load(X)
    total = tl.reduce(value, op="add", axis="n")
    maximum = tl.reduce(value, op="max", axis="n")
    minimum = tl.reduce(value, op="min", axis="n")
    tl.store(Y[0], total)
    tl.store(Y[1], maximum)
    tl.store(Y[2], minimum)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def product_sum(X: tl.View[tl.f32, (N,)], Weight: tl.View[tl.f32, (N,)],
                Y: tl.View[tl.f32, (1,)], *, BLOCK: tl.constexpr):
    total = tl.f32(0.0)
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        product = tl.load(X[part]) * tl.load(Weight[part])
        total += tl.reduce(product, op="add", axis="n")
    tl.store(Y[0], total)


@triton.jit(model="weft", alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def dot_f16(X: tl.View[tl.f16, (N,)], Weight: tl.View[tl.f16, (N,)],
            Y: tl.View[tl.f32, (1,)], *, BLOCK: tl.constexpr):
    total = tl.f32(0.0)
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        total += tl.reduce_dot(tl.f32(tl.load(X[part])), tl.f32(tl.load(Weight[part])), over="n")
    tl.store(Y[0], total)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def mixed_dot_f16(X: tl.View[tl.f32, (N,)], Weight: tl.View[tl.f16, (N,)],
                  Y: tl.View[tl.f32, (1,)], *, BLOCK: tl.constexpr):
    total = tl.f32(0.0)
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        total += tl.reduce_dot(tl.load(X[part]), tl.f32(tl.load(Weight[part])), over="n")
    tl.store(Y[0], total)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def mixed_dot_bf16(X: tl.View[tl.f32, (N,)], Weight: tl.View[tl.bf16, (N,)],
                   Y: tl.View[tl.f32, (1,)], *, BLOCK: tl.constexpr):
    total = tl.f32(0.0)
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        total += tl.reduce_dot(tl.load(X[part]), tl.f32(tl.load(Weight[part])), over="n")
    tl.store(Y[0], total)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def transformed_statistics(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, 3)],
                           shift: tl.f32, scale: tl.f32, *, N: tl.constexpr, ROWS: tl.constexpr):
    with tl.level.blocks(M, extent=tl.auto("ROWS")) as rows:
        with tl.level.blocks(N, extent=tl.auto("N")) as columns:
            transformed = (tl.load(X[rows, columns]) - shift) * scale
            squared = transformed * transformed
            total = tl.reduce(squared, op="add", axis="n")
            maximum = tl.reduce(squared, op="max", axis="n")
            minimum = tl.reduce(squared, op="min", axis="n")
            tl.store(Y[rows, 0], total)
            tl.store(Y[rows, 1], maximum)
            tl.store(Y[rows, 2], minimum)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def dynamic_statistics(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, 3)],
                       shift: tl.f32, scale: tl.f32):
    value = (tl.load(X) - shift) * scale
    squared = value * value
    total = tl.reduce(squared, op="add", axis="n")
    maximum = tl.reduce(squared, op="max", axis="n")
    minimum = tl.reduce(squared, op="min", axis="n")
    tl.store(Y[:, 0], total)
    tl.store(Y[:, 1], maximum)
    tl.store(Y[:, 2], minimum)
