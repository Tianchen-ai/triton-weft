# Weft compiler core inside Triton

This component contains Canonical Kernel IR, shared analyses and transforms,
RISC-V RVV/IME and x86 SIMD/AMX physical compilation, and the native C/C++ runtime.

The Python language, JIT, autotuner and native backend are built and installed
through the enclosing Triton tree. Use `@triton.jit(model="weft")` and
`kernel[grid](...)`; this directory has no separate Python distribution or CLI.

Start with `lib/Dialect/Kernel` for source contracts, `lib/Analysis` and
`lib/Transforms` for common mechanisms, and `lib/Target/{RISCV,X86}` for each
target's Backend, Platform, Analysis, Conversion, Transforms and Emission modules.
The [source map](../../docs/weft/development/upstream.md) explains these boundaries
and why RISC-V has custom physical dialect operations while x86 reuses MLIR dialects.

See the [handbook](../../docs/weft/index.md), [installation](../../docs/weft/getting-started/installation.md)
and [native SDK](../../docs/weft/compiler/native-sdk.md). Weft code retains its
[Apache-2.0 license](LICENSE); the enclosing Triton tree retains its own notices.
