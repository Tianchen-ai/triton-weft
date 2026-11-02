from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from .iq2_xxs_staged import _compute


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq2_xxs_staged(
    W: tl.View[ggml_formats.IQ2_XXS, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[ggml_formats.I8X8, (256, 8)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    Y: tl.View[tl.f32, (M, N)],
    *, NC: tl.constexpr, KC: tl.constexpr, MC: tl.constexpr, MR: tl.constexpr, NR: tl.constexpr,
):
    _compute(W, X, Xq, grid, signs, Y, True, NC, KC, MC, MR, NR)
