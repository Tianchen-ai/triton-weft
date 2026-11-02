from __future__ import annotations

import triton
import triton.language as tl
from .activation import silu_value


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def batched_matmul(X: tl.View[tl.f32, (B, M, K)], Weight: tl.View[tl.f32, (B, K, N)],
                   Y: tl.View[tl.f32, (B, M, N)]):
    tl.store(Y, tl.reduce_dot(tl.load(X), tl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def matmul(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (K, N)],
           Y: tl.View[tl.f32, (M, N)], *, MC: tl.constexpr, NC: tl.constexpr):
    with tl.level.tiles(M, extent=tl.auto("MC")) as rows:
        with tl.level.tiles(N, extent=tl.auto("NC")) as columns:
            value = tl.dot(tl.load(X[rows]), tl.load(Weight[:, columns]), over="k")
            tl.store(Y[rows, columns], value)


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Bias": "inputs", "Y": "output"})
def matmul_bias(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (K, N)],
                Bias: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)], *, MC: tl.constexpr, NC: tl.constexpr):
    with tl.level.tiles(M, extent=tl.auto("MC")) as rows:
        with tl.level.tiles(N, extent=tl.auto("NC")) as columns:
            value = tl.dot(tl.load(X[rows]), tl.load(Weight[:, columns]), over="k")
            tl.store(Y[rows, columns], value + tl.load(Bias[columns]))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def linear_silu(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
                 Y: tl.View[tl.f32, (M, N)], *, MC: tl.constexpr, NC: tl.constexpr):
    with tl.level.tiles(M, extent=tl.auto("MC")) as rows:
        with tl.level.tiles(N, extent=tl.auto("NC")) as columns:
            value = tl.dot(tl.load(X[rows]), tl.load(Weight[columns]), over="k")
            tl.store(Y[rows, columns], silu_value(value))


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Bias": "inputs", "Y": "output"})
def linear_bias_silu(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
                      Bias: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)], *, MC: tl.constexpr, NC: tl.constexpr):
    with tl.level.tiles(M, extent=tl.auto("MC")) as rows:
        with tl.level.tiles(N, extent=tl.auto("NC")) as columns:
            value = tl.dot(tl.load(X[rows]), tl.load(Weight[columns]), over="k") + tl.load(Bias[columns])
            tl.store(Y[rows, columns], silu_value(value))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def linear(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
            Y: tl.View[tl.f32, (M, N)], *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    if M == 1:
        with tl.level.cols(N, group=1) as out:
            tl.store(Y[0, out], tl.reduce_dot(tl.load(Weight[out]), tl.load(X[0]), over="k"))
    else:
        with tl.level.tiles(N, extent=tl.auto("NC")) as columns:
            with tl.level.tiles(M, extent=tl.auto("MC")) as rows:
                with tl.level.cols(columns, group=tl.auto("NR")) as out:
                    with tl.level.rows(rows, group=tl.auto("MR")) as row:
                        value = tl.dot(tl.load(X[row]), tl.load(Weight[out]), over="k")
                        tl.store(Y[row, out], value)


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Bias": "inputs", "Y": "output"})
def linear_bias(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
                 Bias: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)], *, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr):
    if M == 1:
        with tl.level.cols(N, group=1) as out:
            tl.store(Y[0, out], tl.reduce_dot(tl.load(Weight[out]), tl.load(X[0]), over="k") + tl.load(Bias[out]))
    else:
        with tl.level.tiles(N, extent=tl.auto("NC")) as columns:
            with tl.level.tiles(M, extent=tl.auto("MC")) as rows:
                with tl.level.cols(columns, group=tl.auto("NR")) as out:
                    with tl.level.rows(rows, group=tl.auto("MR")) as row:
                        value = tl.dot(tl.load(X[row]), tl.load(Weight[out]), over="k")
                        tl.store(Y[row, out], value + tl.load(Bias[out]))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def linear_tile(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
                Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr, MROWS: tl.constexpr, NROWS: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MROWS")) as row:
        with tl.level.cols(N, group=tl.auto("NROWS")) as out:
            accumulator = tl.state(tl.f32, [MROWS, NROWS], init=0)
            with tl.level.blocks(K, extent=tl.auto("BLOCK")) as part:
                accumulator += tl.dot(tl.load(X[row, part]), tl.load(Weight[out, part]), over="k")
            tl.store(Y[row, out], accumulator)


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Bias": "inputs", "Y": "output"})
def linear_bias_tile(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
                     Bias: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr, MROWS: tl.constexpr, NROWS: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MROWS")) as row:
        with tl.level.cols(N, group=tl.auto("NROWS")) as out:
            accumulator = tl.state(tl.f32, [MROWS, NROWS], init=0)
            with tl.level.blocks(K, extent=tl.auto("BLOCK")) as part:
                accumulator += tl.dot(tl.load(X[row, part]), tl.load(Weight[out, part]), over="k")
            tl.store(Y[row, out], accumulator + tl.load(Bias[out]))
