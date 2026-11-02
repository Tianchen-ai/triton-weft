from __future__ import annotations

from triton.integrations import ggml_formats
from ..blocks.quantize import q8_k as quant_q8_k
from ..blocks.vec_dot.iq1_s_q8_k import compute
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def projection(W: tl.View[ggml_formats.IQ1_S, (N, K)], X: tl.View[tl.f32, (M, K)],
               Xq: tl.View[ggml_formats.Q8_K, (M, K)],
               grid: tl.View[ggml_formats.I8X8, (2048, 8)], Y: tl.View[tl.f32, (M, N)]):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], compute(W[column], Xq[row], grid))
