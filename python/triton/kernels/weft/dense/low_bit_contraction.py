from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class PackedU4:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 32
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.encoding
class SignedI8:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 32
    q: tl.i8[32]


@triton.jit(model="weft", alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def low_bit_projection(X: tl.View[SignedI8, (M, K)],
                       Weight: tl.View[PackedU4, (N, K)], Y: tl.View[tl.i32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as rows:
        with tl.level.cols(N, group=tl.auto("NR")) as columns:
            accumulator = tl.state(tl.i32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as part:
                x = tl.load(X[rows, part])
                weight = tl.load(Weight[columns, part])
                accumulator += tl.dot(x.q, weight.q, over="k", acc_dtype=tl.i32)
            tl.store(Y[rows, columns], accumulator)
