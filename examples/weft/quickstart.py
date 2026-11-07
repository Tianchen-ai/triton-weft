"""Compile a complete Weft kernel, inspect its IR and export native execution."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import triton
import triton.language as tl


@triton.jit(model="weft")
def row_sum_squares(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M,)],
                    BLOCK: tl.constexpr = 32):
    with tl.level.rows(M, group=1) as row:
        values = tl.stage(tl.load(X[row]))
        total = tl.state(tl.f32, [1], init=0)
        with tl.level.blocks(N, extent=BLOCK) as block:
            part = values[:, block]
            total += tl.reduce(part * part, axis="n")
        tl.store(Y[row], total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-ir", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    x = np.random.default_rng(19).normal(size=(4, 65)).astype(np.float32)
    y = np.empty(4, dtype=np.float32)
    compiled = row_sum_squares[(1,)](x, y, BLOCK=32, num_cpu_threads=1)
    reference = np.sum(x.astype(np.float64) ** 2, axis=1)
    np.testing.assert_allclose(y, reference, rtol=1e-5, atol=1e-5)
    print("Row sums of squares: complete output matches the reference", y)
    print("Compilation stages:", ", ".join(compiled.asm))
    if args.dump_ir:
        args.dump_ir.mkdir(parents=True, exist_ok=True)
        for name in ("weftir", "x86ir", "riscvir", "llir", "c"):
            if name in compiled.asm:
                (args.dump_ir / ("row_sum_squares." + name)).write_text(str(compiled.asm[name]))
        print("Generated code:", args.dump_ir)
    if args.export:
        compiled.export(args.export)
        print("Native artifact:", args.export)
    return compiled


if __name__ == "__main__":
    main()
