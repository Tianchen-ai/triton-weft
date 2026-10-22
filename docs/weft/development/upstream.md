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
| `third_party/weft/lib/Dialect`, `lib/Analysis` | Model semantics and common analysis without Python or framework objects |
| `third_party/weft/lib/Target/RISCV`, `Target/X86` | Target-specific physical decisions and native emission |
| `third_party/weft/lib/Runtime` | Native ABI, library lifetime and worker capability checks; no MLIR/Python dependency |
| `python/triton/integrations`, `integrations/llama.cpp` | Optional application contracts and thin GGML hooks |
| `cpu.py`, `tools/cpu.py` | Target construction, homogeneous CPU discovery and explicit process launch |
| `CMakeLists.txt`, `setup.py` | One LLVM/MLIR SDK, one `triton` distribution and bundled native toolchain |

The normal GPU routes and upstream tests remain in the fork. Weft verification,
optimized kernel programs, baseline recipes and performance outputs live in
[Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench).
Compiler code must not import or dispatch by names in that corpus.

The extension changes language and compiler frontend interfaces, so a backend
plugin alone cannot add it to an unmodified upstream wheel. Install this fork's
`triton` wheel. The independently organized Canonical/target libraries keep a
future frontend integration from duplicating the physical compiler.

Upstream Triton keeps its MIT license. Weft additions retain Apache-2.0;
the root NOTICE and `third_party/weft/licenses` preserve third-party attribution.
