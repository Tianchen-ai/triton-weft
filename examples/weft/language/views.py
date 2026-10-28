"""Use typed Views, runtime extents and a subview for the final program.

From the repository root: python -m examples.weft.language.views --size 4097
Use TRITON_CACHE_DIR to select the application's compilation cache.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import triton
import triton.language as tl


@triton.jit(model="weft")
def affine_grid(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                scale: tl.f32 = 1.0, *, bias: tl.f32 = 0.0, B: tl.constexpr = 64):
    start = tl.program_id(0) * B
    if start < N:
        # A View carries N; this subview describes just the valid final region.
        count = tl.minimum(B, N - start)
        x = tl.subview(X, offsets=(start,), extents=(count,))
        y = tl.subview(Y, offsets=(start,), extents=(count,))
        tl.store(y, tl.load(x) * scale + bias)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4097)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    x = np.arange(arguments.size, dtype=np.float32)
    y = np.empty_like(x)
    options = {"num_cpu_threads": arguments.threads}
    if arguments.toolchain is not None:
        options["toolchain"] = json.loads(arguments.toolchain.read_text())
    compiled = affine_grid[(triton.cdiv(x.size, 64),)](x, y, 1.25, bias=-0.5, B=64, **options)
    print("View extent:", x.size, "programs:", triton.cdiv(x.size, 64))
    print("Y = 1.25 * X - 0.5:", y[:8])
    if arguments.output is not None:
        print(compiled.export(arguments.output))


if __name__ == "__main__":
    main()
