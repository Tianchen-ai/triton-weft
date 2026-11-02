from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def ime_i8_contract(
    A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i8, (K, N)], C: tl.View[tl.i32, (M, N)]
, *, KB: tl.constexpr, MR: tl.constexpr, NR: tl.constexpr):
    with tl.level.rows(M, group=tl.auto("MR")) as mb:
        with tl.level.cols(N, group=tl.auto("NR")) as nb:
            acc = tl.state(tl.i32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=tl.auto("KB")) as kb:
                acc += tl.dot(tl.load(A[mb, kb]), tl.load(B[kb, nb]), over="k", acc_dtype=tl.i32)
            tl.store(C[mb, nb], acc)
