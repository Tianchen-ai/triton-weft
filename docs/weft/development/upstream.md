# Upstream relationship and extension boundaries

This repository is a fork of [triton-lang/triton](https://github.com/triton-lang/triton),
based on its `v3.8.0` release. `main` contains the product. Upstream history remains
available; the old standalone Weft development history is not imported.

```sh
git remote add upstream https://github.com/triton-lang/triton.git
git diff v3.8.0 -- python/triton/runtime python/triton/compiler python/triton/language
git diff v3.8.0 -- CMakeLists.txt setup.py python/src
```

| Seam | Responsibility |
|---|---|
| `language/core.py`, `language/semantic.py` | Preserve standard builtins and pass explicit CPU contraction constraints through the semantic interface |
| `compiler/code_generator.py`, `compiler/weft/` | Shared Python AST, scopes and SSA; Encoding/Level analysis and Canonical builder |
| `runtime/jit.py`, `runtime/autotuner.py` | Per-model target selection, specialization, common cache and author configuration tuning |
| `backends/weft/` | Target options, physical pipelines, toolchain stage and CPU driver |
| `python/src/weft.cc` | In-process Canonical/physical bindings and native capability query |
| `third_party/weft/lib/Dialect/Kernel/IR` | Canonical types, Encoding, control, memory and numerical contracts, split by semantic responsibility |
| `third_party/weft/lib/Dialect/Kernel/Transforms` | Source specialization before either target lowering |
| `third_party/weft/lib/Analysis` | Shared integer, logical-axis, index, storage, lifetime and dependence analysis |
| `third_party/weft/lib/Transforms` | Shared SCF full-tile transforms and bounded software-pipeline expansion |
| `third_party/weft/lib/Target/RISCV`, `Target/X86` | Separate Backend, Platform, Analysis, Conversion, Transforms and Emission modules |
| `third_party/weft/lib/Runtime` | Native ABI, library lifetime and worker capability checks; no MLIR/Python dependency |
| `python/triton/integrations`, `integrations/llama.cpp` | Optional application contracts and thin GGML hooks |
| `cpu.py`, `tools/cpu.py` | Target construction, homogeneous CPU discovery and explicit process launch |
| `CMakeLists.txt`, `setup.py` | One LLVM/MLIR SDK, one `triton` distribution and bundled native toolchain |

The normal GPU routes and upstream tests remain in the fork. Weft verification,
optimized kernel programs, baseline recipes and performance outputs live in
[Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench).
Compiler code must not import or dispatch by names in that corpus.

Within each target, `Backend` assembles the pipeline, `Platform` describes ISA and
execution capabilities, `Analysis` interprets that target's physical representation,
`Conversion` imports Canonical operations, and `Transforms` selects and materializes
the physical program. Its subdirectories group memory, arithmetic, contraction,
control and resources. `Emission` translates decisions already present in that
program. Target-private headers use paths relative to that target, and public
headers mirror `include/Weft/Target/RISCV` and `include/Weft/Target/X86`.

`Dialect/RISCV/IR` defines the custom RVV/IME physical types and operations.
The x86 physical program uses upstream `vector`, `memref`, `scf`, `arith` and LLVM
operations; `Target/X86/IR/Dialects.cpp` registers those dialects and their memory
effects. There is no second x86 operation vocabulary or empty dialect wrapper.
This follows Triton's separation of `lib/Analysis`, dialect IR/transforms,
conversion and terminal translation, while retaining the CPU physical model.

Canonical shape/axis queries live in `Dialect/Kernel/IR/TypeQueries` and are shared
by verification and lowering. Canonical non-wrapping index proofs live in
`Analysis/IndexRelations`; they do not depend on x86 registers. In contrast,
RVV register groups and x86 packet pressure remain target analyses. A component
belongs in the shared layer because of the facts it consumes, not its file count.

The extension changes language and compiler frontend interfaces, so a backend
plugin alone cannot add it to an unmodified upstream wheel. Install this fork's
`triton` wheel. The independently organized Canonical/target libraries keep a
future frontend integration from duplicating the physical compiler.

Upstream Triton keeps its MIT license. Weft additions retain Apache-2.0;
the root NOTICE and `third_party/weft/licenses` preserve third-party attribution.
