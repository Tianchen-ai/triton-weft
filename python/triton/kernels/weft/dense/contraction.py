from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def tiled_i8(X: tl.View[tl.i8, (M, K)], W: tl.View[tl.i8, (N, K)],
             Y: tl.View[tl.i32, (M, N)], *, MR: tl.constexpr,
             NR: tl.constexpr, KB: tl.constexpr):
    with tl.level.rows(M, group=MR) as rows:
        with tl.level.cols(N, group=NR) as columns:
            acc = tl.state(tl.i32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=KB) as block:
                acc += tl.dot(tl.load(X[rows, block]), tl.load(W[columns, block]),
                              over="k", acc_dtype=tl.i32)
            tl.store(Y[rows, columns], acc)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def dense_projection(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
                     Y: tl.View[tl.f32, (M, N)]):
    tl.store(Y, tl.dot(tl.load(X), tl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Bias": "input", "Y": "output"})
def dense_projection_bias(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
                          Bias: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (M, N)], scale: tl.f32):
    value = tl.dot(tl.load(X), tl.load(Weight), over="k")
    tl.store(Y, value * scale + tl.load(Bias))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def renamed_projection(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (N, K)],
                       Y: tl.View[tl.f32, (M, N)]):
    left = tl.reshape(tl.load(X), shape=(M, K), axes=("m", "feature"), order=("m", "k"))
    right = tl.reshape(tl.load(Weight), shape=(N, K), axes=("n", "feature"), order=("n", "k"))
    tl.store(Y, tl.dot(left, right, over="feature"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def row_major_matmul(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (K, N)],
                     Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as row:
        with tl.level.cols(N, group=tl.auto("NR")) as column:
            tl.store(Y[row, column], tl.dot(tl.load(X[row]), tl.load(Weight[:, column]), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def row_major_matmul_f16(X: tl.View[tl.f16, (M, K)], Weight: tl.View[tl.f16, (K, N)],
                         Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as row:
        with tl.level.cols(N, group=tl.auto("NR")) as column:
            tl.store(Y[row, column], tl.dot(tl.load(X[row]), tl.load(Weight[:, column]), over="k", acc_dtype=tl.f32))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def batched_inner_product(X: tl.View[tl.f32, (B, P, K)],
                          Weight: tl.View[tl.f32, (B, P, K)], Y: tl.View[tl.f32, (B,)], *, ROWS: tl.constexpr):
    with tl.level.blocks(B, extent=tl.auto("ROWS")) as batch:
        value = tl.reduce_dot(tl.load(X[batch]), tl.load(Weight[batch]), over=("p", "k"))
        tl.store(Y[batch], value)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def pairwise_inner_product(X: tl.View[tl.f32, (M, P, K)],
                           Weight: tl.View[tl.f32, (N, P, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as row:
        with tl.level.cols(N, group=tl.auto("NR")) as column:
            value = tl.reduce_dot(tl.load(X[row]), tl.load(Weight[column]), over=("p", "k"))
            tl.store(Y[row, column], value)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def dot_f64(X: tl.View[tl.f64, (K,)], Weight: tl.View[tl.f64, (K,)],
             Y: tl.View[tl.f64, (1,)]):
    tl.store(Y[0], tl.reduce_dot(tl.load(X), tl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "weight", "Y": "output"})
def snapshot_matmul(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (K, N)],
                    Y: tl.View[tl.f32, (M, N)], *, M: tl.constexpr, N: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("M")) as row:
        with tl.level.cols(N, group=tl.auto("N")) as column:
            saved = tl.stage(tl.load(Weight[:, column]))
            tl.store(Weight[:, column], saved * 0.0)
            tl.store(Y[row, column], tl.dot(tl.load(X[row]), saved, over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def dynamic_matmul(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (K, N)],
                   Y: tl.View[tl.f32, (M, N)]):
    tl.store(Y, tl.dot(tl.load(X), tl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def lhs_lane_matmul(X: tl.View[tl.f32, (K, M)], Weight: tl.View[tl.f32, (N, K)],
                    Y: tl.View[tl.f32, (M, N)]):
    tl.store(Y, tl.dot(tl.load(X), tl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def output_lane_f64(X: tl.View[tl.f64, (M, K)], Weight: tl.View[tl.f64, (K, N)],
                    Y: tl.View[tl.f64, (M, N)]):
    tl.store(Y, tl.dot(tl.load(X), tl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def matmul_cast(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (K, N)],
                Y: tl.View[tl.f64, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as row:
        with tl.level.cols(N, group=tl.auto("NR")) as column:
            value = tl.dot(tl.load(X[row]), tl.load(Weight[:, column]), over="k")
            tl.store(Y[row, column], tl.f64(value))


@triton.jit(model="weft", strided=("X", "Weight", "Y", "Norm"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output", "Norm": "norm"})
def matmul_observers(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (K, N)],
                     Y: tl.View[tl.f32, (M, N)], Norm: tl.View[tl.f32, (M,)], *, M: tl.constexpr, N: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("M")) as row:
        with tl.level.cols(N, group=tl.auto("N")) as column:
            value = tl.dot(tl.load(X[row]), tl.load(Weight[:, column]), over="k")
            tl.store(Y[row, column], value)
            tl.store(Norm[row], tl.reduce(value * value, op="add", axis="n"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def batched_matmul(X: tl.View[tl.f32, (B, M, K)], Weight: tl.View[tl.f32, (B, K, N)],
                   Y: tl.View[tl.f32, (B, M, N)]):
    tl.store(Y, tl.reduce_dot(tl.load(X), tl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def batched_lhs_lane(X: tl.View[tl.f32, (B, K, M)], Weight: tl.View[tl.f32, (B, N, K)],
                     Y: tl.View[tl.f32, (B, M, N)]):
    tl.store(Y, tl.reduce_dot(tl.load(X), tl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Gate", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Bias": "input", "Gate": "input", "Y": "output"})
def matmul_epilogue(X: tl.View[tl.f32, (M, K)], Weight: tl.View[tl.f32, (K, N)],
                    Bias: tl.View[tl.f32, (N,)], Gate: tl.View[tl.f32, (M,)],
                    Y: tl.View[tl.f32, (M, N)], scale: tl.f32, *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as row:
        with tl.level.cols(N, group=tl.auto("NR")) as column:
            value = tl.dot(tl.load(X[row]), tl.load(Weight[:, column]), over="k")
            left = value + tl.load(Bias[column])
            right = value * tl.load(Gate[row])
            tl.store(Y[row, column], left * right * scale)


@triton.jit(model="weft", strided=("X", "Weight0", "Weight1", "Bias0", "Bias1", "Y0", "Y1"),
             alias_groups={"X": "input", "Weight0": "input", "Weight1": "input",
                           "Bias0": "input", "Bias1": "input", "Y0": "first", "Y1": "second"})
def paired_projections(X: tl.View[tl.f32, (M, K)], Weight0: tl.View[tl.f32, (K, N)],
                       Weight1: tl.View[tl.f32, (K, N)], Bias0: tl.View[tl.f32, (N,)],
                       Bias1: tl.View[tl.f32, (N,)], Y0: tl.View[tl.f32, (M, N)],
                       Y1: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as row:
        with tl.level.cols(N, group=tl.auto("NR")) as column:
            first = tl.dot(tl.load(X[row]), tl.load(Weight0[:, column]), over="k")
            tl.store(Y0[row, column], first + tl.load(Bias0[column]))
            second = tl.dot(tl.load(X[row]), tl.load(Weight1[:, column]), over="k")
            tl.store(Y1[row, column], second + tl.load(Bias1[column]))
