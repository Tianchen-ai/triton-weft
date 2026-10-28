# Canonical Kernel IR

Canonical IR is the typed representation of the author's numerical program.
It is built in-process by Triton's common frontend and the Weft semantic builder.
Use `compiled.asm["weftir"]` to inspect it.

## What it records

| Source concept | Canonical information |
|---|---|
| View and Encoding | Logical shape, field mapping, access and alias contract |
| Value | Element type, signedness, shape and axis identities |
| Level | Domain, partition, initialization regions, body and carried results |
| state/stage | Birth scope, initialization and legal consumers |
| Arithmetic | Operation boundaries, grouped widths, conversion and reduction axes |
| Memory | Storage provenance, offsets, masks, fill values and ordered effects |
| Contraction | Typed operands, free/reduction axes and the author's `engine` constraint |

An Encoding field keeps its coordinate relation to storage. A grouped partial
keeps its accumulator type. A staged Value keeps its lifetime and consumers.
These facts are available to both target backends and the framework interface.

## Verification and analysis

The verifier checks type and axis agreement, initialization visibility,
carry/handoff types, storage mappings and operation-specific numerical contracts.
Shared analyses derive integer facts, index relations, storage bounds, aliases
and dependence. Rewriting passes rebuild facts invalidated by their changes.

Ordinary Triton tensor semantics enter through the same frontend. Standard dot
preserves its positional matrix convention; explicit named-axis operations retain
their logical identities.

## Implementation

Canonical definitions live in `third_party/weft/include/Weft/Dialect/Kernel`.
`lib/Dialect/Kernel/IR` separates types, Encoding, control, memory and numerical
verification. `lib/Analysis` holds analyses shared by RISC-V and x86.

[Complete operation and verifier specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/dsl/canonical-ir.md).
