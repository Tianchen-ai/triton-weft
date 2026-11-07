---
hide:
  - toc
---

# Triton–Weft 1.0

**Compose expert CPU kernels. Let the compiler construct their machine execution.**

An operator interface leaves many performance choices open: expanding packed
weights or sharing a prepared region, grouping local products, and keeping state
across updates. Each choice has its own scope. CPU kernel authors need to change
and combine them without reconstructing their machine execution every time.

[![Paper Fig. 1: from an operator's implementation choices to a Weft program and native CPU execution](assets/paper/overview.svg)](assets/paper/overview.svg)

*Weft fills this abstraction gap: the program states the implementation
relationships, and the compiler constructs the machine mapping.*

Weft expresses stored fields, typed local arithmetic, shared preparation and
initialized state as composable source objects. Implemented as a Triton
extension, it preserves those choices through Canonical IR and builds vector or
matrix execution for **x86 and RISC-V**. The same model supports quantized and
dense projections, normalization, online attention and streaming statistics.

[Install](getting-started/installation.md) · [Write your first kernel](getting-started/quickstart.md)
· [Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench)
· [Complete paper results](https://tianchen-ai.github.io/triton-weft-bench/paper/gallery/)

## Start with a complete program

[Install Weft](getting-started/installation.md) and run the
[quickstart](getting-started/quickstart.md). It uses typed Views, a staged input
and a state updated across blocks. Continue with complete Q4_K projection,
online attention, Welford statistics and GEMM in the
[runnable examples](getting-started/examples.md).

| Your next task | Read |
|---|---|
| Express storage and numerical choices | [Programming model](programming-model/index.md), [Encoding](dsl/encoding-and-artifacts.md), [operations](dsl/functions-and-operations.md) |
| Compose sharing and state | [Level and state](dsl/values-and-levels.md), [model programs](dsl/programs.md) |
| Select cores and execution engines | [Targets and engines](getting-started/hardware.md) |
| Tune and deploy | [JIT and autotune](compiler/runtime.md), [native C/C++](compiler/native-sdk.md), [llama.cpp](integrations/llama-cpp.md) |
| Reproduce measurements | [Bench workflow](experiments/reproduce.md), [results and data](experiments/results.md) |
| Develop the compiler | [Pipeline](compiler/index.md), [passes](compiler/passes.md), [source map](development/upstream.md) |

Use the language selector for the corresponding English or Chinese chapter.
Both editions use `import triton.language as tl`.

## From source relationships to native execution

[![Triton frontend, Canonical IR and native x86/RISC-V compilation](assets/paper/compiler.svg)](assets/paper/compiler.svg)

The author writes `@triton.jit(model="weft")` and launches with
`kernel[grid](...)`. Typed fields, numerical groups, sharing and state remain
explicit during compilation. Target passes choose vector layouts, matrix
fragments and local storage. The runtime supports JIT caching, autotuning,
native export and application calls.

## Measured results

Across **543 matched comparisons** from 553 operator/target cases, generated
kernels achieve **1.32–1.83×** per-target geometric-mean speedups over optimized
GGML and oneDNN references. The complete distributions show the performance
across encoded arithmetic, format conversion, dense computation and row kernels.

[![Cumulative distributions of all matched cases](assets/paper/operators-ecdf.svg)](assets/paper/operators-ecdf.svg)

| Platform | Matched comparisons | Geometric mean speedup |
|---|---:|---:|
| SG2044 | 132 | **1.60×** |
| K1 | 136 | **1.54×** |
| K3 A100 | 135 | **1.83×** |
| Xeon Platinum 8470Q | 140 | **1.32×** |

Complete inference covers eight model/platform deployments and 18 phase
comparisons. SG2044 uses 64 application threads; K1, K3 and x86 use eight.
K3's CPU-vector comparison accompanies its vendor IME2 deployment.

[![Complete prefill and decode measurements](assets/paper/models.svg)](assets/paper/models.svg)

Performance follows from the source controls and the compiler's use of them.
The [selected results](experiments/results.md) show five-route control tasks,
format reuse and physical mechanisms under fixed source semantics.
**[The complete Bench gallery](https://tianchen-ai.github.io/triton-weft-bench/paper/gallery/)**
contains every paper result figure, from hardware microkernels and thread scaling
to source increments and compiler studies, with links to all sixteen CSV tables.

[Reproduce with Bench](experiments/reproduce.md): select a workload, installed
kernel, reference and CPU. Bench prepares shared inputs, checks outputs and
measures the matching native calls or complete llama.cpp stages.

## Version and implementation

Weft **1.0.0** extends Triton **3.8.0**. The distribution version is
`3.8.0+weft.1.0.0`; the Python API exposes both identities.
See the [release guide](development/release.md) for source packages and
verification, and the [design specifications](https://github.com/Tianchen-ai/triton-weft/tree/main/docs/specification)
for detailed language and compiler contracts.
