"""Reuse a staged row across ordered updates of one state.

Run from the repository root: python -m examples.weft.language.levels
The row Level owns the saved input and accumulator. The inner block Level owns
each update; leaving it carries state back to the row before the final store.
"""
from __future__ import annotations

import numpy as np
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_sum_squares(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M,)],
                     B: tl.constexpr = 32):
    with tl.level.rows(M, group=1) as row:
        values = tl.stage(tl.load(X[row]))
        total = tl.state(tl.f32, [1], init=0)
        with tl.level.blocks(N, extent=B) as block:
            part = values[:, block]
            total += tl.reduce(part * part, axis="n", op="add")
        tl.store(Y[row], total)


def main():
    x = np.linspace(-1, 1, 4 * 65, dtype=np.float32).reshape(4, 65)
    y = np.empty(4, dtype=np.float32)
    compiled = row_sum_squares[(1,)](x, y)
    print("Sum of squares:", y)
    print("Compilation stages:", list(compiled.asm))


if __name__ == "__main__":
    main()
