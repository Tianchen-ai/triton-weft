# Triton–Weft handbook

Write CPU kernels in Triton, then express the storage, numerical grouping and
reuse that matter to your implementation. Weft adds these relationships to the
language and carries them through compilation to RISC-V and x86.

## A path through the handbook

| Your next task | Read |
|---|---|
| Run a first kernel | [Install](getting-started/installation.md), then [quickstart](getting-started/quickstart.md) |
| Understand the added programming model | [Model](programming-model/index.md), [Encoding](dsl/encoding-and-artifacts.md), [Level and state](dsl/values-and-levels.md) |
| Choose cores, ISA and matrix arithmetic | [Targets and engines](getting-started/hardware.md) |
| Tune and deploy | [JIT and autotune](compiler/runtime.md), [C++](compiler/native-sdk.md), [llama.cpp](integrations/llama-cpp.md) |
| Reproduce the paper | [Bench workflow](experiments/reproduce.md), [figures and measured setup](experiments/results.md) |
| Develop the compiler | [Compiler pipeline](compiler/index.md), [passes](compiler/passes.md), [extension guide](development/extensions.md) |

The language selector switches between corresponding English and Chinese pages.
Code uses `import triton.language as tl` in both editions.

[![Shared frontend and target compilation](assets/paper/compiler.png)](assets/paper/compiler.png)

## From source to execution

1. Write a normal Triton kernel with `@triton.jit(model="weft")`.
2. Add View, Encoding or Level where your algorithm needs explicit relationships.
3. Launch with `kernel[grid](...)`; Triton specializes, compiles, caches and executes.
4. Inspect `CompiledKernel.asm`, tune finite configurations, or export a native artifact.

The [examples](getting-started/examples.md) build this workflow one concept at a
time. Each provides input preparation, a callable kernel and a host application.

## Paper results

Reproduce the paper's operator and model experiments with Triton–Weft and
[Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench).
The dataset covers 553 operator–platform cases and nine model configurations.

[![Representative kernels and all matched operator comparisons](assets/paper/operators.png)](assets/paper/operators.png)

| Platform | Matched native comparisons | Geometric mean speedup |
|---|---:|---:|
| SG2044 | 132 | **1.60×** |
| K1 | 136 | **1.55×** |
| K3 A100 | 135 | **1.82×** |
| Xeon Platinum 8470Q | 140 | **1.56×** |

Speedup is the matching reference time divided by Triton–Weft time.

[![All nine model configurations, with prefill and decode separated](assets/paper/models.png)](assets/paper/models.png)

The 18 model stages cover Llama 2 7B, DeepSeek 8B and Qwen 2.5 14B with eight
threads. K3 CPU–RVV and vendor IME2 use their respective reference policies.
[Reproduce these results](experiments/reproduce.md) or open the
[figures, full CSV tables and measurement settings](experiments/results.md).

## Documentation and source

This handbook explains usage and compiler responsibilities. Detailed numerical,
IR and lifecycle contracts are maintained in the
[design specifications](https://github.com/Tianchen-ai/triton-weft/tree/main/docs/specification).
The [source map](development/upstream.md) connects those contracts to their
implementation. [Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench)
owns comparison workloads, reference libraries and experiment results.
