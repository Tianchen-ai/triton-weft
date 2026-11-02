from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from ...quantize.x86.blocks32 import quantize
from ...vec_dot.x86.blocks32 import compute


@triton.jit
def quantize_rows(X, Xq, Sums, M, K, AFFINE: tl.constexpr, SAVE_SUM: tl.constexpr):
    for row in range(M):
        if SAVE_SUM:
            quantize(X[row], Xq[row], K, AFFINE, 4, True, Sums[row])
        else:
            quantize(X[row], Xq[row], K, AFFINE, 4)


@triton.jit
def project_row_tiles(W, Xq, Y, M, N, K, CENTER: tl.constexpr,
                      HIGH: tl.constexpr, AFFINE: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr,
                      SIGNED: tl.constexpr = False, Sums=None,
                      DECODE: tl.constexpr = None, SCALE: tl.constexpr = None, Table=None, Scale=None):
    with tl.level.rows(M, group=BM) as rows:
        with tl.level.cols(N, group=BN) as columns:
            partial = tl.state(tl.f32, [BN, BM], init=0)
            correction = tl.state(tl.f32, [BN, BM], init=0)
            with tl.level.blocks(K, extent=32) as block:
                w, x = W[columns, block], Xq[rows, block]
                if HIGH:
                    q = tl.u8(tl.load(w.q)) | (tl.u8(tl.load(w.qh)) << 4)
                else:
                    q = tl.load(w.q)
                if DECODE is not None:
                    q = DECODE(q, Table)
                if SIGNED:
                    q = tl.u8(q) ^ tl.u8(128)
                elif CENTER:
                    q = tl.u8(q)
                weight = tl.reshape(q, (BN, 32), axes=("N", "k"), order=("N", "K"))
                activation = tl.reshape(tl.load(x.q), (BM, 32), axes=("M", "k"), order=("M", "K"))
                products = tl.reduce_dot(weight, activation, over="k", acc_dtype=tl.i32, engine="vector")
                if CENTER or SIGNED:
                    products -= tl.i32(CENTER + (128 if SIGNED else 0)) * tl.i32(tl.load(Sums[rows, block].value))
                if SCALE is not None:
                    wd = SCALE(w, Scale)
                else:
                    wd = tl.f32(tl.load(w.d))
                partial += (wd * tl.f32(tl.load(x.d))) * tl.f32(products)
                if AFFINE:
                    correction += tl.f32(tl.load(w.m)) * tl.f32(tl.load(x.s))
            value = partial + correction
            tl.store(Y[rows, columns], tl.reshape(value, (BM, BN), axes=("M", "N"), order=("M", "N")))


@triton.jit
def project(W, X, Xq, Y, M, N, K, CENTER: tl.constexpr,
            HIGH: tl.constexpr, AFFINE: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, SIGNED: tl.constexpr = False, Sums=None,
            DECODE: tl.constexpr = None, SCALE: tl.constexpr = None, Table=None, Scale=None):
    quantize_rows(X, Xq, Sums, M, K, AFFINE, CENTER or SIGNED)
    # Single-row calls keep a vector output. Batched calls put columns in
    # adjacent lanes and reuse one weight cohort across the activation rows.
    if BM == 1:
        project_row_tiles(W, Xq, Y, M, N, K, CENTER, HIGH, AFFINE, BM, BN, SIGNED, Sums, DECODE, SCALE, Table, Scale)
    else:
        with tl.level.cols(N, group=BN) as columns:
            with tl.level.rows(M, group=BM) as rows:
                partial = tl.state(tl.f32, [BM, BN], init=0)
                correction = tl.state(tl.f32, [BM, BN], init=0)
                with tl.level.blocks(K, extent=32) as block:
                    w, x = W[columns, block], Xq[rows, block]
                    if HIGH:
                        q = tl.u8(tl.load(w.q)) | (tl.u8(tl.load(w.qh)) << 4)
                    else:
                        q = tl.load(w.q)
                    if DECODE is not None:
                        q = DECODE(q, Table)
                    if SIGNED:
                        q = tl.u8(q) ^ tl.u8(128)
                    weight = tl.reshape(q, (BN, 32), axes=("N", "k"), order=("N", "K"))
                    activation = tl.reshape(tl.load(x.q), (BM, 32), axes=("M", "k"), order=("M", "K"))
                    products = tl.reduce_dot(activation, weight, over="k", acc_dtype=tl.i32, engine="vector")
                    if CENTER or SIGNED:
                        products -= tl.i32(CENTER + (128 if SIGNED else 0)) * tl.i32(tl.load(Sums[rows, block].value))
                    if SCALE is not None:
                        wd = SCALE(w, Scale)
                    else:
                        wd = tl.f32(tl.load(w.d))
                    partial += (tl.f32(tl.load(x.d)) * wd) * tl.f32(products)
                    if AFFINE:
                        correction += tl.f32(tl.load(x.s)) * tl.f32(tl.load(w.m))
                value = partial + correction
                tl.store(Y[rows, columns], tl.reshape(value, (BM, BN), axes=("M", "N"), order=("M", "N")))


@triton.jit
def project_natural(W, X, Xq, Y, M, N, K, CENTER: tl.constexpr,
                    HIGH: tl.constexpr, AFFINE: tl.constexpr, GROUP: tl.constexpr = 4):
    for row in range(M):
        quantize(X[row], Xq[row], K, AFFINE, 4)
        for column in range(N):
            value = compute(W[column], Xq[row], K, CENTER, HIGH, AFFINE, GROUP)
            tl.store(Y[row, column], value)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_0(W: tl.View[ggml_formats.Q4_0, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project_natural(W, X, Xq, Y, M, N, K, 8, False, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_1(W: tl.View[ggml_formats.Q4_1, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project_natural(W, X, Xq, Y, M, N, K, 0, False, True)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_0(W: tl.View[ggml_formats.Q5_0, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project_natural(W, X, Xq, Y, M, N, K, 16, True, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_1(W: tl.View[ggml_formats.Q5_1, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_1, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project_natural(W, X, Xq, Y, M, N, K, 0, True, True)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q8_0(W: tl.View[ggml_formats.Q8_0, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)]):
    project_natural(W, X, Xq, Y, M, N, K, 0, False, False, 2)
