"""Author-declared storage layouts used by the affine rows kernels."""
from __future__ import annotations

import triton
import triton.language as tl
from triton.integrations import ggml_formats


@triton.derive
def Q4KRows(W: tl.View[ggml_formats.Q4_K, (N, K)], *, rows: tl.static[int]) -> tl.View[Q4KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "sc": 1, "m": 1, "q": 4})


@triton.derive
def Q5KRows(W: tl.View[ggml_formats.Q5_K, (N, K)], *, rows: tl.static[int]) -> tl.View[Q5KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "sc": 6, "m": 6, "qh": 4, "q": 4})


@triton.derive
def IQ4XSRows(W: tl.View[ggml_formats.IQ4_XS, (N, K)], *, rows: tl.static[int]) -> tl.View[IQ4XSRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "scales_h": 2, "scales_l": 4, "q": 4})


@triton.derive
def Q2KRows(W: tl.View[ggml_formats.Q2_K, (N, K)], *, rows: tl.static[int]) -> tl.View[Q2KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "scales": 4, "q": 4})
