from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class IME2Weight8:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f16
    q: tl.i8[32]


@triton.encoding
class IME2Activation8:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f32
    neg_sum: tl.i16
    q: tl.i8[32]


@triton.derive
def IME2Weight8Rows(W: tl.View[IME2Weight8, (N, K)], *, rows: tl.static[int]) -> tl.View[IME2Weight8Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 32})


@triton.derive
def IME2Activation8Rows(X: tl.View[IME2Activation8, (M, K)], *, rows: tl.static[int]) -> tl.View[IME2Activation8Rows[rows], (M, K)]:
    return tl.interleave(X, rows=rows, field_chunks={"d": 4, "neg_sum": 2, "q": 32})


@triton.jit
def packed_i8(W, Xq, Y, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=MR) as rows:
        with tl.level.cols(N, group=NR) as columns:
            result = tl.state(tl.f32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as block:
                w = tl.load(W[columns, block])
                x = tl.load(Xq[rows, block])
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
