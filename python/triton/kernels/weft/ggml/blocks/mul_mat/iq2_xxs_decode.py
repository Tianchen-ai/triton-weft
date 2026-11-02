from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
from ..vec_dot import iq2_xxs_q8_k as vd_iq2_xxs_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq2_xxs_decode(
    W: tl.View[ggml_formats.IQ2_XXS, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[ggml_formats.I8X8, (256, 8)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    Y: tl.View[tl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vd_iq2_xxs_q8_k.compute(W[column], Xq[row], grid, signs)
            tl.store(Y[row, column], value)
