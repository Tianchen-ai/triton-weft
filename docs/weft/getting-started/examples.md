# Runnable examples

Run these examples from the repository root, after installing this Triton
distribution on a supported RISC-V or x86 host. Examples import the installed
Triton package; no source overlay or Bench package is required. NumPy examples
need NumPy; the PyTorch example additionally needs a supported CPU PyTorch build
and the [framework source seam](installation.md#可选应用依赖).

## Ordinary Triton

```sh
python examples/weft/triton_basics.py --operation affine
python examples/weft/triton_basics.py --operation softmax
python examples/weft/triton_basics.py --operation matmul
python examples/weft/triton_basics.py --operation scan
python examples/weft/triton_basics.py --operation histogram --threads 2
```

The callable programs are included in `examples/weft/triton_basics.py`. The application
allocates input/output storage and launches them through `kernel[grid]`.

| Example | Language and execution concepts |
|---|---|
| Affine | Runtime scalars, constexpr block size, masked tails |
| Row softmax | Strides, max/sum, stable exponential normalization |
| Matmul | Two-dimensional grid, broadcast addresses, accumulator `tl.dot`, K loop |
| Scan | Inclusive `tl.cumsum`, per-row grid, masked input/output |
| Histogram | Many programs update shared bins with `tl.atomic_add` |

Histogram input indices must lie in the allocated bin range. Atomic operations
serialize conflicting updates; they do not create a barrier across the grid.
The application initializes the count array before launch.

## Tune and deploy the selected kernel

```sh
python examples/weft/autotune_export.py /path/to/new/artifact
```

This selects a source block size using `triton.Config` and `triton.autotune`,
exports the selected `CompiledKernel`, then loads and calls that native artifact.
The export directory must not already exist. Loaded calls use the fixed source
configuration and need neither the original DSL definition nor a compiler.
For plain JIT without tuning, `python examples/weft/use.py --output /path/to/new/artifact`
uses the existing View-based affine program.

## PyTorch

```sh
python -m triton.tools.pytorch
python examples/weft/pytorch_cpu.py
```

The example first passes CPU tensors to a hand-written Triton kernel, then asks
Inductor to generate a normalization kernel with `cpu_backend="triton"`.
Keep shape, stride and mutation semantics in the normal tensor workflow.
Gradient programs use explicit backward registration, as in
the Bench corpus's `weft_bench.kernels.dense.torch_affine`; there is no implicit kernel autodifferentiation.

## Weft model features and larger programs

| Topic | Source to read | Contract |
|---|---|---|
| View and subviews | `examples/weft/use.py` | Runtime extents and source-owned data partitioning |
| Encoding and Level | `examples/weft/llama_cpp.py` | Packed fields, finite-width reduction and block traversal |
| Matrix/vector selection | `examples/weft/contraction.py` | Per-dot engine constraint and actual artifact extensions |
| Attention and model programs | Bench `weft_bench.kernels.programs` | State, workspace and ordered host composition |
| llama.cpp | [Deployment tutorial](../integrations/llama-cpp.md) | Exported library, validated role binding and existing GGML workers |

These sources describe numerical programs, not a promise that every target
implements every possible configuration. A required extension must be present
and actually used by the resulting program. Inspect `asm["weftir"]`,
`asm["riscvir"]` and `asm["c"]` to understand the selected implementation.

Numerical verification and performance measurements belong to the independent
[Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench), which consumes these same
programs. The example scripts do not carry a separate benchmark runner.
