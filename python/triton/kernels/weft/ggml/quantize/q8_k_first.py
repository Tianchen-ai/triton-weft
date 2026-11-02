from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit(model="weft")
def q8_k_quantize_first_absolute(
    X: tl.View[tl.f32, (K,)], Y: tl.View[ggml_formats.Q8_K, (K,)]
):
    """Quantize finite inputs using the first absolute maximum and RNE."""
    with tl.level.blocks(K, extent=256) as kb:
        block = tl.load(X[kb])
        maximum = tl.reduce(block, op="max")
        minimum = tl.reduce(block, op="min")
        extreme = minimum
        if tl.abs(maximum) > tl.abs(minimum):
            extreme = maximum
        elif maximum > tl.f32(0) and maximum == -minimum:
            # Equal opposite extrema retain GGML's first occurrence rule.
            magnitude = tl.abs(block)
            positions = tl.arange(0, 256, dtype=tl.u32, axis="k")
            first = tl.reduce(tl.select(magnitude == maximum, positions, tl.u32(256)), op="min")
            extreme = tl.load(X[kb][first])
        inverse = tl.f32(0.0)
        d = tl.f32(0.0)
        if extreme != tl.f32(0.0):
            inverse = tl.f32(-127.0) / extreme
            d = tl.f32(1.0) / inverse
        q = tl.narrow(block * inverse, tl.i8, rounding="rne", saturation=True)
        groups = tl.reshape(q, (16, 16), axes=("group", "lane"), order=("K",))
        totals = tl.reduce(tl.widen(groups, tl.i16), op="add", axis="lane")
        tl.store(Y[kb].q, q)
        tl.store(Y[kb].bsum, tl.reshape(totals, (16,), axes=("K",), order=("group",)))
        tl.store(Y[kb].ds, d)
