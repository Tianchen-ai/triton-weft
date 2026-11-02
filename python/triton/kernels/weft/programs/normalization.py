from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Weight": "parameters"})
def rms_norm(X: tl.View[tl.f32, (M, N)], Weight: tl.View[tl.f32, (N,)],
             Y: tl.View[tl.f32, (M, N)], epsilon: tl.f32 = 1e-5, *, BLOCK: tl.constexpr,
             ACCUM_F64: tl.constexpr = False):
    for row in range(M):
        row_value = tl.load(X[row])
        if ACCUM_F64:
            squares = tl.reduce(tl.f64(row_value * row_value), op="add", axis="n")
            mean = tl.f32(squares / tl.f64(N))
        else:
            squares = tl.reduce(row_value * row_value, op="add", axis="n")
            mean = squares / tl.f32(N)
        inverse = tl.rsqrt(mean + epsilon)
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.load(X[row, part])
            weight = tl.load(Weight[part])
            tl.store(Y[row, part], value * inverse * weight)


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "activation", "Y": "activation",
                           "Weight": "parameters", "Bias": "parameters"})
def layer_norm(X: tl.View[tl.f32, (M, N)], Weight: tl.View[tl.f32, (N,)],
               Bias: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)],
               epsilon: tl.f32 = 1e-5, *, BLOCK: tl.constexpr):
    for row in range(M):
        total = tl.reduce(tl.load(X[row]), op="add", axis="n")
        mean = total / tl.f32(N)
        centered = tl.load(X[row]) - mean
        squares = tl.reduce(centered * centered, op="add", axis="n")
        inverse = tl.rsqrt(squares / tl.f32(N) + epsilon)
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.load(X[row, part])
            weight = tl.load(Weight[part])
            bias = tl.load(Bias[part])
            tl.store(Y[row, part], (value - mean) * inverse * weight + bias)


@triton.jit(model="weft", strided=("X", "Residual", "Weight", "Y"),
             alias_groups={"X": "activation", "Residual": "activation",
                           "Y": "activation", "Weight": "parameters"})
def residual_rms_norm(X: tl.View[tl.f32, (M, N)], Residual: tl.View[tl.f32, (M, N)],
                      Weight: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)],
                      epsilon: tl.f32 = 1e-5, *, BLOCK: tl.constexpr):
    for row in range(M):
        row_value = tl.load(X[row]) + tl.load(Residual[row])
        squares = tl.reduce(row_value * row_value, op="add", axis="n")
        inverse = tl.rsqrt(squares / tl.f32(N) + epsilon)
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.load(X[row, part]) + tl.load(Residual[row, part])
            tl.store(Y[row, part], value * inverse * tl.load(Weight[part]))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Weight": "parameters"})
def rms_norm_row(X: tl.View[tl.f32, (M, N)], Weight: tl.View[tl.f32, (N,)],
                 Y: tl.View[tl.f32, (M, N)], epsilon: tl.f32 = 1e-5, *, N: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("N")) as columns:
            value = tl.load(X[row, columns])
            squares = tl.reduce(value * value, op="add", axis="n")
            inverse = tl.rsqrt(squares / tl.f32(N) + epsilon)
            tl.store(Y[row, columns], value * inverse * tl.load(Weight[columns]))


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "activation", "Y": "activation",
                           "Weight": "parameters", "Bias": "parameters"})
def layer_norm_row(X: tl.View[tl.f32, (M, N)], Weight: tl.View[tl.f32, (N,)],
                   Bias: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)],
                   epsilon: tl.f32 = 1e-5, *, N: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("N")) as columns:
            value = tl.load(X[row, columns])
            mean = tl.reduce(value, op="add", axis="n") / tl.f32(N)
            centered = value - mean
            squares = tl.reduce(centered * centered, op="add", axis="n")
            inverse = tl.rsqrt(squares / tl.f32(N) + epsilon)
            tl.store(Y[row, columns], centered * inverse * tl.load(Weight[columns]) + tl.load(Bias[columns]))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Weight": "parameters"})
def rms_norm_tile(X: tl.View[tl.f32, (M, N)], Weight: tl.View[tl.f32, (N,)],
                  Y: tl.View[tl.f32, (M, N)], epsilon: tl.f32 = 1e-5, *, N: tl.constexpr, ROWS: tl.constexpr):
    with tl.level.blocks(M, extent=tl.auto("ROWS")) as rows:
        with tl.level.blocks(N, extent=tl.auto("N")) as columns:
            value = tl.load(X[rows, columns])
            squares = tl.reduce(value * value, op="add", axis="n")
            inverse = tl.rsqrt(squares / tl.f32(N) + epsilon)
            tl.store(Y[rows, columns], value * inverse * tl.load(Weight[columns]))


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "activation", "Y": "activation",
                           "Weight": "parameters", "Bias": "parameters"})
def layer_norm_tile(X: tl.View[tl.f32, (M, N)], Weight: tl.View[tl.f32, (N,)],
                    Bias: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)],
                    epsilon: tl.f32 = 1e-5, *, N: tl.constexpr, ROWS: tl.constexpr):
    with tl.level.blocks(M, extent=tl.auto("ROWS")) as rows:
        with tl.level.blocks(N, extent=tl.auto("N")) as columns:
            value = tl.load(X[rows, columns])
            mean = tl.reduce(value, op="add", axis="n") / tl.f32(N)
            centered = value - mean
            squares = tl.reduce(centered * centered, op="add", axis="n")
            inverse = tl.rsqrt(squares / tl.f32(N) + epsilon)
            tl.store(Y[rows, columns], centered * inverse * tl.load(Weight[columns]) + tl.load(Bias[columns]))
