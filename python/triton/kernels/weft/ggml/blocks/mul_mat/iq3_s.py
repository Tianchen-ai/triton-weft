from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
from ..vec_dot import iq3_s_q8_k as vd_iq3_s_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq3_s(
    W: tl.View[ggml_formats.IQ3_S, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[ggml_formats.I8X4, (512, 4)],
    Y: tl.View[tl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row_base in range(0, M, 16):
        for column_base in range(0, N, 16):
            for row in range(row_base, tl.minimum(M, row_base + 16)):
                for column in range(column_base, tl.minimum(N, column_base + 16)):
                    tl.store(Y[row, column], vd_iq3_s_q8_k.compute(W[column], Xq[row], grid))
