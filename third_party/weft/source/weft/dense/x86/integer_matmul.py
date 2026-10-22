from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def full_reduction(X: tl.View[tl.i8, (M, K)], W: tl.View[tl.i8, (N, K)],
                   Y: tl.View[tl.i32, (M, N)], *, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, BM: tl.constexpr = 16, BN: tl.constexpr = 16):
    """Reuse one activation snapshot while keeping each modular dot closed."""
    with tl.level.rows(M, group=BM) as rows:
        activation = tl.load(X[rows, :])
        with tl.level.cols(N, group=BN) as columns:
            weights = tl.load(W[columns, :])
            result = tl.reduce_dot(activation, weights, over="K", acc_dtype=tl.i32)
            tl.store(Y[rows, columns], result)
