from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", strided=("X", "Y"))
def affine_strided(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)],
                   scale: tl.f32 = 1.0, *, bias: tl.f32 = 0.0, B: tl.constexpr):
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=tl.auto("B")) as block:
            tl.store(Y[row, block], tl.load(X[row, block]) * scale + bias)
