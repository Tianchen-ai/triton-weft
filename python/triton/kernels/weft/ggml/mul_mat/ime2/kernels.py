from __future__ import annotations

import triton
import triton.language as tl
from triton.kernels.weft.ggml.layouts.ime2 import IME2Weight8, IME2Activation8, IME2Weight4Affine, IME2Weight4AffineRows, IME2Weight8Rows, IME2Activation8Rows


@triton.jit
def packed_i8(W, Xq, Y, MR: tl.constexpr, NR: tl.constexpr,
              CLOSED_K: tl.constexpr = False):
    with tl.level.rows(M, group=MR) as rows:
        with tl.level.cols(N, group=NR) as columns:
            result = tl.state(tl.f32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as block:
                w = tl.load(W[columns, block])
                x = tl.load(Xq[rows, block])
                if CLOSED_K:
                    integer = tl.dot(x.q, w.q, over="k", acc_dtype=tl.i32)
                else:
                    integer = tl.state(tl.i32, [MR, NR], init=0)
                    with tl.level.subtiles(block, extent=16) as part:
                        integer += tl.dot(x.q[part], w.q[part], over="k", acc_dtype=tl.i32)
                if MR == 1:
                    scale = tl.reshape(tl.f32(w.d) * x.d, (MR, NR),
                                       axes=("m", "n"), order=("m", "n"))
                    result += tl.f32(integer) * scale
                else:
                    result += (tl.f32(integer) * tl.f32(w.d)) * x.d
            tl.store(Y[rows, columns], result)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i8_prefill(W: tl.View[IME2Weight8Rows[32], (N, K)],
               Xq: tl.View[IME2Activation8Rows[4], (M, K)],
               Y: tl.View[tl.f32, (M, N)], *, NR: tl.constexpr = 32):
    packed_i8(W, Xq, Y, 4, NR)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i8_decode(W: tl.View[IME2Weight8Rows[32], (N, K)],
              Xq: tl.View[IME2Activation8Rows[1], (M, K)],
              Y: tl.View[tl.f32, (M, N)], *, NR: tl.constexpr = 32):
    packed_i8(W, Xq, Y, 1, NR)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i8_prefill_closed(W: tl.View[IME2Weight8Rows[32], (N, K)],
                      Xq: tl.View[IME2Activation8Rows[4], (M, K)],
                      Y: tl.View[tl.f32, (M, N)], *, NR: tl.constexpr = 32):
    packed_i8(W, Xq, Y, 4, NR, True)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i8_prefill_rows8(W: tl.View[IME2Weight8Rows[32], (N, K)],
                     Xq: tl.View[IME2Activation8Rows[4], (M, K)],
                     Y: tl.View[tl.f32, (M, N)], *, NR: tl.constexpr = 32):
    packed_i8(W, Xq, Y, 8, NR, True)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i8_decode_closed(W: tl.View[IME2Weight8Rows[32], (N, K)],
                     Xq: tl.View[IME2Activation8Rows[1], (M, K)],
                     Y: tl.View[tl.f32, (M, N)], *, NR: tl.constexpr = 32):
    packed_i8(W, Xq, Y, 1, NR, True)


@triton.jit
def packed_i4_affine(W, Xq, Y, M, MR: tl.constexpr, NR: tl.constexpr = 32,
                     NATIVE_I4: tl.constexpr = False):
    with tl.level.rows(M, group=MR) as rows:
        with tl.level.cols(N, group=NR) as columns:
            result = tl.state(tl.f32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as block:
                w = tl.load(W[columns, block])
                x = tl.load(Xq[rows, block])
                if NATIVE_I4:
                    high = tl.dot(tl.i4(x.q >> 4), w.q, over="k", acc_dtype=tl.i32)
                    low = tl.dot(tl.u4(x.q), w.q, over="k", acc_dtype=tl.i32)
                    integer = high * 16 + low
                else:
                    integer = tl.dot(x.q, tl.i8(w.q), over="k", acc_dtype=tl.i32)
                integer += tl.i32(x.neg_sum) * tl.i32(w.zp)
                if MR == 1:
                    scale = tl.reshape(tl.f32(w.d) * x.d, (MR, NR),
                                       axes=("m", "n"), order=("m", "n"))
                    result += tl.f32(integer) * scale
                else:
                    result += (tl.f32(integer) * tl.f32(w.d)) * x.d
            tl.store(Y[rows, columns], result)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i4_affine_prefill(W: tl.View[IME2Weight4AffineRows[32], (N, K)],
                     Xq: tl.View[IME2Activation8Rows[4], (M, K)],
                     Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr = 8,
                     NR: tl.constexpr = 32):
    packed_i4_affine(W, Xq, Y, M, MR, NR)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i4_affine_decode(W: tl.View[IME2Weight4AffineRows[32], (N, K)],
                    Xq: tl.View[IME2Activation8Rows[1], (M, K)],
                    Y: tl.View[tl.f32, (M, N)]):
    packed_i4_affine(W, Xq, Y, M, 1)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i4_affine_native_prefill(W: tl.View[IME2Weight4AffineRows[32], (N, K)],
                            Xq: tl.View[IME2Activation8Rows[4], (M, K)],
                            Y: tl.View[tl.f32, (M, N)], *, MR: tl.constexpr = 8,
                            NR: tl.constexpr = 32):
    packed_i4_affine(W, Xq, Y, M, MR, NR, True)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i4_affine_native_tile(W: tl.View[IME2Weight4AffineRows[32], (N, K)],
                         Xq: tl.View[IME2Activation8Rows[4], (M, K)],
                         Y: tl.View[tl.f32, (M, N)], *, M: tl.constexpr = 8,
                         MR: tl.constexpr = 8, NR: tl.constexpr = 32):
    packed_i4_affine(W, Xq, Y, M, MR, NR, True)


@triton.jit(model="weft", strided=("Y",), alias_groups={"Y": "output"})
def i4_affine_half_decode(W: tl.View[IME2Weight4AffineRows[32], (N, K)],
                          Xq: tl.View[IME2Activation8Rows[1], (M, K)],
                          Y: tl.View[tl.f32, (M, N)], *, NR: tl.constexpr = 32):
    """M1 HP contract: two half-rounded nibble contributions per block."""
    with tl.level.rows(M, group=1) as rows:
        with tl.level.cols(N, group=NR) as columns:
            result = tl.state(tl.f32, [1, NR], init=0)
            with tl.level.blocks(K, extent=32) as block:
                w, x = tl.load(W[columns, block]), tl.load(Xq[rows, block])
                high = tl.dot(tl.i4(x.q >> 4), w.q, over="k", acc_dtype=tl.i32)
                low = tl.dot(tl.u4(x.q), w.q, over="k", acc_dtype=tl.i32)
                high_half = tl.f16(tl.f32(high) * tl.f32(16))
                product = tl.f16(tl.f32(high_half) + tl.f32(low))
                correction = tl.f16(tl.i16(x.neg_sum) * tl.i16(w.zp))
                corrected = tl.f32(product) + tl.f32(correction)
                result += (corrected * tl.f32(w.d)) * x.d
            tl.store(Y[rows, columns], result)
