"""Shared GGML storage declarations and explicit author layout derivation."""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.weft.declarations import EncodingDefinition
from weft_core.formats.ggml import CATALOG

globals().update({name: EncodingDefinition.from_schema(schema) for name, schema in CATALOG.items()})


@triton.derive
def Q4K_I(
    W: tl.View[Q4_K, (M, K)], *, rows: tl.static[int]
) -> tl.View[Q4K_I[rows], (M, K)]:
    return tl.interleave(W, rows=rows)
