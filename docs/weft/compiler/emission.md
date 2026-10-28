# From physical operations to native code

Emission translates verified physical decisions into the target toolchain's input.
The generated code is inspectable through `CompiledKernel.asm` and exported artifacts.

## RISC-V

RVV operations become typed intrinsics; selected extension operations can use
typed local assembly. Layout, active length, conversion, local packing,
software-pipeline versions and resource decisions have already been materialized.
Clang 23 compiles the resulting C with the selected ISA, ABI and VLEN.

Inspect `asm["riscvir"]`, then `asm["c"]`. For a load, follow the logical region
to its physical access and active-domain mask. For a contraction, follow the
chosen local operation and accumulator representation.

## x86

The x86 pipeline lowers its physical vector/memory program through LLVM dialect
conversion and translates it to LLVM IR. LLVM's x86 code generator selects final
machine registers and instructions for the declared feature set.

Inspect `asm["x86ir"]`, then `asm["llir"]`. SIMD width and AMX use appear in
the selected operations and artifact requirements.

## Native result

Both paths produce a shared library, typed ABI metadata and runtime dependencies.
Triton loads the result for JIT or exports it for [C++](native-sdk.md) and
[llama.cpp](../integrations/llama-cpp.md).

[Complete emission specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/emission.md).
