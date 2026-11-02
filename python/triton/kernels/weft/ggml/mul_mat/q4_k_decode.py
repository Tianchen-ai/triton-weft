from __future__ import annotations

from triton.integrations import ggml_formats
from ..blocks.quantize import q8_k as quant_q8_k
import triton
import triton.language as tl
from ..vec_dot.q4_k_q8_k import compute


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection(
    W: tl.View[ggml_formats.Q4_K, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
):
    """Full decode projection using the same grouped dot as the leaf entry."""
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], compute(W[column], Xq[row]))
