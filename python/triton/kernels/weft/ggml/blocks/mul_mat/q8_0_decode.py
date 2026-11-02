from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_0 as quant_q8_0
from ..vec_dot import q8_0_q8_0 as vd_q8_0_q8_0
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_q8_0_decode(
    W: tl.View[ggml_formats.Q8_0, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_0, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
):
    quant_q8_0.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], vd_q8_0_q8_0.compute(W[column], Xq[row]))
