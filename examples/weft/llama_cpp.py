"""Export a Q8_0 dot with the exact GGML single-row calling convention.

python examples/weft/llama_cpp.py /path/to/new-package --profile k1 --toolchain toolchain.json
This tutorial expresses the complete arithmetic; optimized variants live in Bench.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton import cpu
from triton.integrations.ggml_formats import Q8_0
from triton.integrations.llama_cpp import create_package


@triton.jit(model="weft")
def q8_dot(W: tl.View[Q8_0, (K,)], X: tl.View[Q8_0, (K,)], Y: tl.View[tl.f32, (1,)]):
    total = tl.f32(0.0)
    with tl.level.blocks(K, extent=32) as block:
        w = tl.load(W[block])
        x = tl.load(X[block])
        integer = tl.reduce_dot(w.q, x.q, over="k", acc_dtype=tl.i32, engine="vector")
        total += tl.f32(integer) * (tl.f32(w.d) * tl.f32(x.d))
    tl.store(Y[0], total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", default="native", choices=["native", *cpu.profiles()])
    parser.add_argument("--toolchain", type=Path)
    args = parser.parse_args()
    options = {"toolchain": json.loads(args.toolchain.read_text())} if args.toolchain else {}
    compiled = triton.compile(ASTSource(q8_dot, {}, {}), target=cpu.target(args.profile), options=options)
    create_package(args.output, {"q8_dot": compiled}, bindings={"ggml.vec_dot": {"q8_0": "q8_dot"}})
    print("Native package:", args.output)


if __name__ == "__main__":
    main()
