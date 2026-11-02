from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", strided=("Grad", "DX"))
def affine_backward(Grad: tl.View[tl.f32, (M, N)], DX: tl.View[tl.f32, (M, N)],
                    scale: tl.f32 = 1.0):
    # Zero-stride broadcast gradients are ordinary read-only dense Views.
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=tl.auto(64)) as block:
            tl.store(DX[row, block], tl.load(Grad[row, block]) * scale)
