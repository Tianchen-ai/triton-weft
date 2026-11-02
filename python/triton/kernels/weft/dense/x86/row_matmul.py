"""Author-owned row packets for row-major CPU projection weights."""
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_matmul(X, W, Y, *, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, INTEGER: tl.constexpr = False):
    """X @ W.T; keep row partials until the complete K traversal finishes.

    This source variant chooses its grouping and accumulation tree explicitly.
    The same code accepts F32 activations with F16 weights, BF16 pairs, or
    integer pairs with I32 products and wrapping sums. No input repack is needed.
    """
    lane = tl.arange(0, BK).to(tl.int64)
    row = tl.arange(0, BN).to(tl.int64)
    for m in range(M):
        for base in range(0, N, BN):
            n = base + row
            if INTEGER:
                acc = tl.full((BN, BK), 0, tl.int32)
            else:
                acc = tl.full((BN, BK), 0.0, tl.float32)
            for start in range(0, K, BK):
                k = start + lane
                x = tl.load(X + m * K + k, k < K, other=0)
                w = tl.load(W + n[:, None] * K + k[None, :],
                            (n[:, None] < N) & (k[None, :] < K), other=0)
                if INTEGER:
                    acc += w.to(tl.int32) * x[None, :].to(tl.int32)
                else:
                    acc += w.to(tl.float32) * x[None, :].to(tl.float32)
            value = tl.sum(acc, 1)
            tl.store(Y + m * N + n, value, n < N)
