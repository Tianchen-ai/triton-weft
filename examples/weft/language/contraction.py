"""Compose a typed contraction with logical output regions and integer state.

python -m examples.weft.language.contraction --engine vector
python -m triton.tools.cpu --profile x86-amx --cpus 0 examples/weft/language/contraction.py --engine matrix
"""
from __future__ import annotations

import argparse
import numpy as np
import triton
import triton.language as tl


@triton.jit(model="weft")
def contract(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i8, (N, K)],
             C: tl.View[tl.i32, (M, N)], ENGINE: tl.constexpr,
             MR: tl.constexpr = 4, NR: tl.constexpr = 4,
             KB: tl.constexpr = 64):
    with tl.level.rows(M, group=MR) as rows:
        with tl.level.cols(N, group=NR) as columns:
            total = tl.state(tl.i32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=KB) as block:
                total += tl.dot(tl.load(A[rows, block]),
                                tl.load(B[columns, block]), over="k",
                                acc_dtype=tl.i32, engine=ENGINE)
            tl.store(C[rows, columns], total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("auto", "matrix", "vector"), default="auto")
    args = parser.parse_args()
    a = (np.arange(16 * 64) % 31 - 15).astype(np.int8).reshape(16, 64)
    b = (np.arange(64 * 16) % 29 - 14).astype(np.int8).reshape(64, 16)
    c = np.empty((16, 16), dtype=np.int32)
    compiled = contract[(1,)](a, np.ascontiguousarray(b.T), c,
                             ENGINE=args.engine, num_cpu_threads=1)
    np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))
    print("Integer contraction: all outputs match the I32 reference")
    print(c[:2, :4])
    print("Actual extensions:", compiled.metadata.kernel_abi["used_extensions"])
    print("Physical IR stages:", [key for key in compiled.asm if key.endswith("ir")])


if __name__ == "__main__":
    main()
