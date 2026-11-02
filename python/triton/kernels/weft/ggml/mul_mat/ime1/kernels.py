from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from triton.kernels.weft.ggml.layouts.ime1 import IMEQ4Centered, IMEQ4Affine, IMEQ8, IMEQ4Centered_I, IMEQ4Affine_I, IMEQ8_I


@triton.jit(model="weft")
def affine_zero_point(d, m):
    ratio = -tl.f32(m) / tl.f32(d)
    clipped = tl.minimum(tl.f32(15.0), tl.maximum(tl.f32(0.0), ratio))
    return tl.narrow(clipped, tl.u8, rounding="dynamic", saturation=True)


@triton.jit(model="weft")
def pack_q4_0(W: tl.View[ggml_formats.Q4_0, (N, K)], P: tl.View[IMEQ4Centered_I[16], (N, K)]):
    for row in range(N):
        with tl.level.blocks(K, extent=32) as kb:
            w = tl.load(W[row, kb])
            tl.store(P[row, kb].d, w.d)
            tl.store(P[row, kb].q, w.q)


@triton.jit(model="weft")
def pack_q4_1(W: tl.View[ggml_formats.Q4_1, (N, K)], P: tl.View[IMEQ4Affine_I[16], (N, K)]):
    for row in range(N):
        with tl.level.blocks(K, extent=32) as kb:
            w = tl.load(W[row, kb])
            tl.store(P[row, kb].d, w.d)
            tl.store(P[row, kb].zp, affine_zero_point(w.d, w.m))
            tl.store(P[row, kb].q, w.q)


@triton.jit(model="weft")
def pack_q4_k(W: tl.View[ggml_formats.Q4_K, (N, K)], P: tl.View[IMEQ4Affine_I[16], (N, K)]):
    for row in range(N):
        with tl.level.blocks(K, extent=256) as kb:
            w = tl.load(W[row, kb])
            with tl.level.subtiles(extent=32) as s:
                d = tl.f16(tl.f32(w.d) * tl.f32(w.sc[s]))
                m = tl.f16(-tl.f32(w.dmin) * tl.f32(w.m[s]))
                tl.store(P[row, s].d, d)
                tl.store(P[row, s].zp, affine_zero_point(d, m))
                tl.store(P[row, s].q, w.q[s])


@triton.jit(model="weft")
def quantize_activation(X, Xq):
    for row in range(M):
        with tl.level.blocks(K, extent=32) as kb:
            x = tl.load(X[row, kb])
            amax = tl.reduce(tl.abs(x), op="max")
            d = amax * tl.f32(0.007874015748031496)
            inverse = tl.f32(0.0)
            if amax != tl.f32(0.0):
                inverse = tl.f32(1.0) / d
            q = tl.narrow(x * inverse, tl.i8, rounding="dynamic", saturation=True)
            tl.store(Xq[row, kb].d, d)
            tl.store(Xq[row, kb].q, q)


@triton.jit(model="weft")
def quantize_ime1_prefill(X: tl.View[tl.f32, (M, K)], Xq: tl.View[IMEQ8_I[4], (M, K)]):
    quantize_activation(X, Xq)


@triton.jit(model="weft")
def mul_mat_centered(W, X, Xq, Y, *, MR: tl.constexpr, NR: tl.constexpr):
    quantize_activation(X, Xq)
    packed_mat_centered(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def packed_mat_centered(W, Xq, Y, *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as mb:
        with tl.level.cols(N, group=tl.auto("NR")) as nb:
            acc = tl.state(tl.f32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as kb:
                x = tl.load(Xq[mb, kb])
                w = tl.load(W[nb, kb])
                partial = tl.state(tl.i32, [MR, NR], init=0)
                with tl.level.subtiles(extent=8) as s:
                    q = tl.i8(w.q[s]) - tl.i8(8)
                    partial += tl.dot(x.q[s], q, over="k", acc_dtype=tl.i32)
                scale = x.d * tl.f32(w.d)
                acc += tl.f32(partial) * scale
            tl.store(Y[mb, nb], acc)


@triton.jit(model="weft")
def mul_mat_affine(W, X, Xq, Y, *, MR: tl.constexpr, NR: tl.constexpr):
    quantize_activation(X, Xq)
    packed_mat_affine(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def packed_mat_affine(W, Xq, Y, *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as mb:
        with tl.level.cols(N, group=tl.auto("NR")) as nb:
            acc = tl.state(tl.f32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as kb:
                x = tl.load(Xq[mb, kb])
                w = tl.load(W[nb, kb])
                partial = tl.state(tl.i32, [MR, NR], init=0)
                with tl.level.subtiles(extent=8) as s:
                    q = tl.i8(w.q[s]) - tl.i8(w.zp)
                    partial += tl.dot(x.q[s], q, over="k", acc_dtype=tl.i32)
                scale = x.d * tl.f32(w.d)
                acc += tl.f32(partial) * scale
            tl.store(Y[mb, nb], acc)


@triton.jit(model="weft")
def q4_centered_decode(W: tl.View[IMEQ4Centered_I[16], (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[IMEQ8_I[1], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    mul_mat_centered(W, X, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def q4_centered_prefill(W: tl.View[IMEQ4Centered_I[16], (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[IMEQ8_I[4], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    mul_mat_centered(W, X, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def q4_affine_decode(W: tl.View[IMEQ4Affine_I[16], (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[IMEQ8_I[1], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    mul_mat_affine(W, X, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def q4_affine_prefill(W: tl.View[IMEQ4Affine_I[16], (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[IMEQ8_I[4], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    mul_mat_affine(W, X, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft", strided=("Y",))
def q4_centered_packed_decode(W: tl.View[IMEQ4Centered_I[16], (N, K)], Xq: tl.View[IMEQ8_I[1], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    packed_mat_centered(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft", strided=("Y",))
def q4_affine_packed_decode(W: tl.View[IMEQ4Affine_I[16], (N, K)], Xq: tl.View[IMEQ8_I[1], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    packed_mat_affine(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft", strided=("Y",))
def q4_affine_packed_decode_correction(W: tl.View[IMEQ4Affine_I[16], (N, K)], Xq: tl.View[IMEQ8_I[1], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    # The IME affine packing contract stores a Q4 zero point in [0, 15].
    # Apply its correction once per block, after the exact integer products.
    with tl.level.rows(M, group=tl.auto("MR")) as mb:
        with tl.level.cols(N, group=tl.auto("NR")) as nb:
            acc = tl.state(tl.f32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as kb:
                x = tl.load(Xq[mb, kb])
                w = tl.load(W[nb, kb])
                partial = tl.state(tl.i32, [MR, NR], init=0)
                with tl.level.subtiles(extent=8) as s:
                    partial += tl.dot(x.q[s], tl.i8(w.q[s]), over="k", acc_dtype=tl.i32)
                total = tl.i32(tl.reduce(tl.i16(x.q), axis="k"))
                corrected = partial - total * tl.i32(w.zp)
                scale = x.d * tl.f32(w.d)
                acc += tl.f32(corrected) * scale
            tl.store(Y[mb, nb], acc)


@triton.jit(model="weft", strided=("Y",))
def q4_centered_packed_decode_block(W: tl.View[IMEQ4Centered_I[16], (N, K)], Xq: tl.View[IMEQ8_I[1], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as mb:
        with tl.level.cols(N, group=tl.auto("NR")) as nb:
            acc = tl.state(tl.f32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as kb:
                x = tl.load(Xq[mb, kb])
                w = tl.load(W[nb, kb])
                partial = tl.dot(x.q, tl.i8(w.q), over="k", acc_dtype=tl.i32, engine="matrix")
                total = tl.i32(tl.reduce(tl.i16(x.q), axis="k"))
                corrected = partial - total * tl.i32(8)
                scale = x.d * tl.f32(w.d)
                acc += tl.f32(corrected) * scale
            tl.store(Y[mb, nb], acc)


@triton.jit(model="weft", strided=("Y",))
def q4_affine_packed_decode_block(W: tl.View[IMEQ4Affine_I[16], (N, K)], Xq: tl.View[IMEQ8_I[1], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as mb:
        with tl.level.cols(N, group=tl.auto("NR")) as nb:
            acc = tl.state(tl.f32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as kb:
                x = tl.load(Xq[mb, kb])
                w = tl.load(W[nb, kb])
                partial = tl.dot(x.q, tl.i8(w.q), over="k", acc_dtype=tl.i32, engine="matrix")
                total = tl.i32(tl.reduce(tl.i16(x.q), axis="k"))
                corrected = partial - total * tl.i32(w.zp)
                scale = x.d * tl.f32(w.d)
                acc += tl.f32(corrected) * scale
            tl.store(Y[mb, nb], acc)


@triton.jit(model="weft", strided=("Y",))
def q4_centered_packed_prefill(W: tl.View[IMEQ4Centered_I[16], (N, K)], Xq: tl.View[IMEQ8_I[4], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    packed_mat_centered(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft", strided=("Y",))
def q4_affine_packed_prefill(W: tl.View[IMEQ4Affine_I[16], (N, K)], Xq: tl.View[IMEQ8_I[4], (M, K)], Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    packed_mat_affine(W, Xq, Y, MR=MR, NR=NR)
