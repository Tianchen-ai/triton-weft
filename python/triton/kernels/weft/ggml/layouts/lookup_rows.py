"""Author-declared storage layouts used by the lookup rows kernels."""
from __future__ import annotations

import triton
import triton.language as tl
from triton.integrations import ggml_formats


@triton.derive
def IQ4Rows(W: tl.View[ggml_formats.IQ4_NL, (N, K)], *, rows: tl.static[int]) -> tl.View[IQ4Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 4})


@triton.derive
def MXRows(W: tl.View[ggml_formats.MXFP4, (N, K)], *, rows: tl.static[int]) -> tl.View[MXRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"e": 1, "q": 4})
