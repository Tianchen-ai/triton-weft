from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def sigmoid_value(value):
    small = tl.exp(-tl.abs(value))
    numerator = tl.select(value >= 0.0, 1.0, small)
    return numerator / (1.0 + small)


@triton.jit
def silu_value(value):
    return value * sigmoid_value(value)


@triton.jit
def softplus_value(value):
    return tl.maximum(value, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(value)))


@triton.jit(model="weft", strided=("X", "Y"))
def exponential(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tl.store(Y[row, part], tl.exp(tl.load(X[row, part])))


@triton.jit(model="weft", strided=("X", "Y"))
def logarithm(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tl.store(Y[row, part], tl.log(tl.load(X[row, part])))


@triton.jit(model="weft", strided=("X", "Y"))
def softplus(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tl.store(Y[row, part], softplus_value(tl.load(X[row, part])))


@triton.jit(model="weft", strided=("X", "Y"))
def silu(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tl.store(Y[row, part], silu_value(tl.load(X[row, part])))


@triton.jit(model="weft", strided=("Gate", "Up", "Y"))
def swiglu(Gate: tl.View[tl.f32, (M, N)], Up: tl.View[tl.f32, (M, N)],
            Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = silu_value(tl.load(Gate[row, part])) * tl.load(Up[row, part])
            tl.store(Y[row, part], value)


@triton.jit(model="weft", strided=("X", "Residual", "Y"))
def add(X: tl.View[tl.f32, (M, N)], Residual: tl.View[tl.f32, (M, N)],
         Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tl.store(Y[row, part], tl.load(X[row, part]) + tl.load(Residual[row, part]))


@triton.jit(model="weft", strided=("X", "Y"))
def copy(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tl.store(Y[row, part], tl.load(X[row, part]))


@triton.jit(model="weft", strided=("Y",))
def fill(Y: tl.View[tl.f32, (M, N)], value: tl.f32 = 0.0, *, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("ROWS")) as row:
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            tile = tl.state(tl.f32, [ROWS, BLOCK], init=value)
            tl.store(Y[row, part], tile)


@triton.jit(model="weft", strided=("Condition", "X", "Other", "Y"),
             alias_groups={"Condition": "condition", "X": "activation", "Other": "activation", "Y": "activation"})
def where(Condition: tl.View[tl.i1, (M, N)], X: tl.View[tl.f32, (M, N)],
           Other: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            condition = tl.load(Condition[row, part])
            selected = tl.load(X[row, part])
            other = tl.load(Other[row, part])
            tl.store(Y[row, part], tl.select(condition, selected, other))
