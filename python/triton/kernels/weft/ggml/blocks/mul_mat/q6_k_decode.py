from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
from ..vec_dot import q6_k_q8_k as vd_q6_k_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_q6_k_decode(
    W: tl.View[ggml_formats.Q6_K, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], vd_q6_k_q8_k.compute(W[column], Xq[row]))
