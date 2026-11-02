from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_0 as quant_q8_0
from ..vec_dot import mxfp4_q8_0 as vd_mxfp4_q8_0
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_mxfp4_decode(
    W: tl.View[ggml_formats.MXFP4, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_0, (M, K)],
    codebook: tl.View[tl.i8, (16,)],
    scale: tl.View[tl.f32, (256,)],
    Y: tl.View[tl.f32, (M, N)],
):
    quant_q8_0.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vd_mxfp4_q8_0.compute(W[column], Xq[row], codebook, scale)
            tl.store(Y[row, column], value)
