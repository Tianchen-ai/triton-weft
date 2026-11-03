# Shared Weft compiler core

This component contains Canonical Kernel IR, shared analyses and transforms,
RISC-V RVV/IME and x86 SIMD/AMX physical compilation, and the native C/C++ runtime.

The `weft-core` distribution contains the compiler, native runtime, neutral
Canonical construction API, shared Encoding schemas and target toolchain.
Triton–Weft and TileLang–Weft each retain their own source frontend, JIT, cache
and autotuner. Both call the same `libweft_compiler` through an opaque C ABI;
LLVM/MLIR objects and symbols remain private to the compiler library.

For a standalone source build, select the LLVM/MLIR SDK matching the enclosing
Triton tree, then build this directory with CMake or its `pyproject.toml`:

```sh
cmake -S . -B build -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" \
  -DWEFT_INSTALL_ROOT=weft_core -DWEFT_BUILD_PYTHON=ON
cmake --build build
cmake --install build --prefix sdk
```

`WEFT_CLANG_ROOT`, `WEFT_OPENMP_LIBRARY` and `WEFT_OPENMP_INCLUDE_DIR` select
bundled native toolchain resources. `WEFT_CORE_SDK` packages an already-built
SDK for the same host architecture and builds the binding for the current
Python ABI. It validates native execution and the compiler ABI before packaging.

Native applications load exported artifacts through `libweft_runtime` and its
typed C/C++ interface. Optional application contracts and llama.cpp hooks live
in `integrations/llama.cpp`; the runtime does not link the compiler or a frontend.

Start with `lib/Dialect/Kernel` for source contracts, `lib/Analysis` and
`lib/Transforms` for common mechanisms, and `lib/Target/{RISCV,X86}` for each
target's Backend, Platform, Analysis, Conversion, Transforms and Emission modules.
The [source map](../../docs/weft/development/upstream.md) explains these boundaries
and why RISC-V has custom physical dialect operations while x86 reuses MLIR dialects.

See the [handbook](../../docs/weft/index.md), [installation](../../docs/weft/getting-started/installation.md)
and [native SDK](../../docs/weft/compiler/native-sdk.md). Weft code retains its
[Apache-2.0 license](LICENSE); the enclosing Triton tree retains its own notices.
