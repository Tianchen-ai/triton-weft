# Add a hardware extension

Weft separates capability facts, local instruction contracts, physical transforms,
emission and runtime checks. Use these interfaces for a RISC-V extension or an
x86 instruction family.

## Machine facts and local contracts

Describe ISA features, operand/accumulator types, supported dimensions, alignment
and resources. A named CPU profile expands to those facts.
Define each local operation's numerical behavior, logical coverage, operand
representation, effects, transfers and temporary resources. Legality checks reuse
shared integer, axis and storage proofs.

The author program supplies traversal, source groups, state and reuse scopes.
An extension implements a local part of that physical program.

## Integrate the pipeline

| Step | Responsibility |
|---|---|
| Capability | Platform description and native discovery |
| Legality | Typed numerical, layout and effect requirements |
| Selection | Respect the contraction engine and finite legal options |
| Materialization | Transfers, packing, conversion and the selected operation |
| Resources | Live registers/fragments, local buffers and schedule |
| Emission | The selected intrinsic, assembly or LLVM operation |
| Runtime | Artifact requirements checked on actual workers |

RISC-V modules live in `Target/RISCV/Extensions/SpacemitIME1` and `SpacemitIME2`.
x86 AMX belongs to its contraction and platform modules.
Common numerical and storage semantics stay in Canonical IR and `lib/Analysis`.

## Run the author program

Write a DSL program using the operation and compile through the ordinary Triton
frontend. Bench checks numerical boundaries, tails and real target execution,
then compares the affected workload with its matched reference.
Inspect actual artifact extensions and observed calls.

[Complete extension contract](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/development/extensions.md) ·
[Source map](upstream.md) · [Bench workflow](../experiments/reproduce.md).
