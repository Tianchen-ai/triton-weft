from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
from ..vec_dot import tq1_0_q8_k_lane16 as vd_tq1_0
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_tq1_0_lane16_decode(
    W: tl.View[ggml_formats.TQ1_0, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    powers: tl.View[tl.u32, (5,)],
    Y: tl.View[tl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], vd_tq1_0.compute(W[column], Xq[row], powers))
