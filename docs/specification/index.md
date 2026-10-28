# Weft design specifications

These documents define the numerical, storage, lifetime, IR and physical
compilation contracts. The [bilingual handbook](https://tianchen-ai.github.io/triton-weft/)
provides the user path and a concise explanation of each subsystem.

| Contract | Specification |
|---|---|
| Author model | [Programming model](programming-model/index.md), [boundaries](programming-model/non-goals.md) |
| Storage and ABI | [Encoding and View](dsl/encoding-and-artifacts.md) |
| Lifecycle | [Level, state, stage and handoff](dsl/values-and-levels.md) |
| Arithmetic | [Functions and operations](dsl/functions-and-operations.md) |
| Semantic IR | [Canonical IR](dsl/canonical-ir.md) |
| Compilation | [Overview](compiler/index.md), [physical machine](compiler/physical-machine.md) |
| RISC-V | [IR](compiler/riscv-ir.md), [passes](compiler/passes.md), [leaves](compiler/leaves.md), [emission](compiler/emission.md) |
| x86 | [Physical compilation](compiler/x86.md) |
| Optimization | [Selection and resource principles](compiler/optimization-principles.md) |
| Execution | [Runtime](compiler/runtime.md), [native SDK](compiler/native-sdk.md) |
| Extensions | [Hardware contracts](development/extensions.md) |
| Validation | [Numerical checks](development/validation.md), [references](development/benchmarking.md) |

The handbook and specifications describe the same model at different levels of
detail. Approved semantic changes update these contracts and their implementations together.
