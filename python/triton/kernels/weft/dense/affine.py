from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def affine(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
           scale: tl.f32 = 1.0, *, bias: tl.f32 = 0.0, B: tl.constexpr = 64):
    with tl.level.blocks(N, extent=tl.auto("B")) as block:
        value = tl.load(X[block])
        tl.store(Y[block], value * scale + bias)


@triton.jit(model="weft")
def affine_grid(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                scale: tl.f32 = 1.0, *, bias: tl.f32 = 0.0, B: tl.constexpr = 64):
    start = tl.program_id(0) * B
    if start < N:
        count = tl.minimum(B, N - start)
        x = tl.subview(X, offsets=(start,), extents=(count,))
        y = tl.subview(Y, offsets=(start,), extents=(count,))
        tl.store(y, tl.load(x) * scale + bias)


@triton.jit(model="weft")
def affine_full_tiles(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                      scale: tl.f32 = 1.0, *, bias: tl.f32 = 0.0,
                      B: tl.constexpr = 256):
    """Keep the full-tile traversal separate from its final partial window."""
    end = N // B * B
    for start in range(0, end, B):
        x = tl.subview(X, offsets=(start,), extents=(B,))
        y = tl.subview(Y, offsets=(start,), extents=(B,))
        tl.store(y, tl.load(x) * scale + bias)
    if end < N:
        x = tl.subview(X, offsets=(end,), extents=(N - end,))
        y = tl.subview(Y, offsets=(end,), extents=(N - end,))
        tl.store(y, tl.load(x) * scale + bias)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output"})
def affine_disjoint(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                    scale: tl.f32 = 1.0, *, bias: tl.f32 = 0.0):
    """A whole-domain pointwise program with independent input and output."""
    tl.store(Y, tl.load(X) * scale + bias)
