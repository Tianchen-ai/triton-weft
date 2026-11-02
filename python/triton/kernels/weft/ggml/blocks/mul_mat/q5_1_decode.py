from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_1 as quant_q8_1
from ..vec_dot import q5_1_q8_1 as vd_q5_1_q8_1
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_q5_1_decode(
    W: tl.View[ggml_formats.Q5_1, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_1, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
):
    quant_q8_1.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], vd_q5_1_q8_1.compute(W[column], Xq[row]))
