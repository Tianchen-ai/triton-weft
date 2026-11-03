# Compiler pipeline

The common Triton frontend constructs a Canonical program with Weft's logical
relationships. Target passes turn that program into an executable physical
representation. The final emitter translates the selected representation.

[![Compiler architecture from the paper](../assets/paper/compiler.svg)](../assets/paper/compiler.svg)

## Follow one kernel

| Stage | Information retained or added | Inspect |
|---|---|---|
| Triton frontend | Types, expressions, control flow, Encoding, Level initialization and SSA carry | Source diagnostics |
| Canonical Kernel IR | Logical axes, numerical widths, storage effects, state/stage ownership | `asm["weftir"]` |
| Shared analysis | Integer facts, axes, index relations, storage spans, alias and lifetime | Physical pass decisions |
| RISC-V physical compilation | Lane/time/replica placement, local memory, RVV/IME operations, schedule and resources | `asm["riscvir"]` |
| x86 physical compilation | SIMD packets, matrix operands, local buffers, AMX resources and LLVM conversion | `asm["x86ir"]` |
| Code generation | Intrinsic C/local assembly or LLVM IR | `asm["c"]` / `asm["llir"]` |
| Native compilation and loading | Shared library, typed ABI and execution requirements | `asm["so"]`, exported manifest |

The two target paths share author semantics and analyses. A target's physical
representation records the decisions needed to run that program on its machine.
For example, a staged panel can become registers or local packed storage; the
source's initialization and sharing scope remain visible.

## Read and develop the implementation

- [Canonical IR](../dsl/canonical-ir.md): the source program's typed contract.
- [Physical representation](physical-machine.md): lanes, packets, fragments and storage.
- [Passes](passes.md): analysis, selection, materialization and verification.
- [RISC-V](riscv-ir.md) and [x86](x86.md): target-specific realization.
- [Code emission](emission.md): translation into native toolchain input.

The [source map](../development/upstream.md) links these roles to directories.
The [complete compiler specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/index.md)
defines the Canonical-to-Physical boundary used during development.
