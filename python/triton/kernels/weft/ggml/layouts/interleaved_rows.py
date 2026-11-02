"""Author-declared storage layouts used by the interleaved rows kernels."""
from __future__ import annotations

import triton
import triton.language as tl
from triton.integrations import ggml_formats


@triton.derive
def Q4_0Rows(W: tl.View[ggml_formats.Q4_0, (N, K)], *, rows: tl.static[int]) -> tl.View[Q4_0Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})


@triton.derive
def Q4_1Rows(W: tl.View[ggml_formats.Q4_1, (N, K)], *, rows: tl.static[int]) -> tl.View[Q4_1Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "m": 2, "q": 4})


@triton.derive
def Q5_0Rows(W: tl.View[ggml_formats.Q5_0, (N, K)], *, rows: tl.static[int]) -> tl.View[Q5_0Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "qh": 4, "q": 4})


@triton.derive
def Q5_1Rows(W: tl.View[ggml_formats.Q5_1, (N, K)], *, rows: tl.static[int]) -> tl.View[Q5_1Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "m": 2, "qh": 4, "q": 4})


@triton.derive
def Q8_0Rows(W: tl.View[ggml_formats.Q8_0, (N, K)], *, rows: tl.static[int]) -> tl.View[Q8_0Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})
