"""Choose a contraction engine while keeping ordinary Triton pointer syntax.

python -m examples.weft.language.contraction --engine vector
python -m triton.tools.cpu --profile x86-amx --cpus 0 examples/weft/language/contraction.py --engine matrix
"""
import argparse
import numpy as np
import triton
import triton.language as tl


@triton.jit(model="weft")
def contract(A, B, C, ENGINE: tl.constexpr):
    m = tl.arange(0, 16)
    n = tl.arange(0, 16)
    k = tl.arange(0, 64)
    a = tl.load(A + m[:, None] * 64 + k[None, :])
    b = tl.load(B + k[:, None] * 16 + n[None, :])
    result = tl.dot(a, b, engine=ENGINE)
    tl.store(C + m[:, None] * 16 + n[None, :], result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("auto", "matrix", "vector"), default="auto")
    args = parser.parse_args()
    a = (np.arange(16 * 64) % 31 - 15).astype(np.int8).reshape(16, 64)
    b = (np.arange(64 * 16) % 29 - 14).astype(np.int8).reshape(64, 16)
    c = np.empty((16, 16), dtype=np.int32)
    compiled = contract[(1,)](a, b, c, ENGINE=args.engine, num_cpu_threads=1)
    print(c[:2, :4])
    print("Actual extensions:", compiled.metadata.kernel_abi["used_extensions"])
    print("Physical IR stages:", [key for key in compiled.asm if key.endswith("ir")])


if __name__ == "__main__":
    main()
