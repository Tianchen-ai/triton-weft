from __future__ import annotations

from triton.integrations import ggml_formats
from ..blocks.quantize import q8_k as quant_q8_k
import triton
import triton.language as tl
from ..vec_dot.ternary import compute_tq2


@triton.jit(noinline=True)
def row_dot(W: tl.View[ggml_formats.TQ2_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
            LANES: tl.constexpr, DEFERRED: tl.constexpr):
    return compute_tq2(W, X, LANES, DEFERRED)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection(W: tl.View[ggml_formats.TQ2_0, (N, K)], X: tl.View[tl.f32, (M, K)],
               Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
               *, LANES: tl.constexpr, DEFERRED: tl.constexpr = False):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], row_dot(W[column], Xq[row], LANES, DEFERRED))
