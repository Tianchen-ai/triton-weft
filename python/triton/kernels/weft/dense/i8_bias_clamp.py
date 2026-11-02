from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def dot_tile(A, B, mb, nb, K, MR: tl.constexpr, NR: tl.constexpr, KB: tl.constexpr):
    acc = tl.state(tl.i32, [MR, NR], init=0)
    with tl.level.blocks(K, extent=tl.auto("KB")) as kb:
        acc += tl.dot(tl.load(A[mb, kb]), tl.load(B[kb, nb]),
                      over="k", acc_dtype=tl.i32)
    return acc


@triton.jit
def bias_clamp(value, bias):
    return tl.minimum(tl.maximum(value + bias, tl.i32(0)), tl.i32(127))


@triton.jit(model="weft")
def i8_bias_clamp(A: tl.View[tl.i8, (M, K)],
                  B: tl.View[tl.i8, (K, N)],
                  Bias: tl.View[tl.i32, (N,)],
                  C: tl.View[tl.i32, (M, N)], *, MR: tl.constexpr, NR: tl.constexpr, KB: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as mb:
        with tl.level.cols(N, group=tl.auto("NR")) as nb:
            result = bias_clamp(dot_tile(A, B, mb, nb, K, MR, NR, KB), tl.load(Bias[nb]))
            tl.store(C[mb, nb], result)
