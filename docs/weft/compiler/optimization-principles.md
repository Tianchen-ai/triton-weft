# Optimize physical work

Begin with the generated program: count repeated loads, decoding, address
calculation, conversions, reductions, spills and waiting. Then identify the source
relationship that makes a better implementation legal.

## A useful investigation order

1. Read the author program's blocking, state and stage placement.
2. Inspect Canonical IR for the intended numerical and reuse relationships.
3. Inspect physical layouts, actual accesses and conversion operations.
4. Read resource conclusions and the native output.
5. Measure the affected workload with Bench's matching native reference.

## Bounded selection

The target supplies finite legal choices such as vector grouping, packet width,
fragment representation and supply strategy. Explicit author or backend settings
fix their respective dimensions. Each candidate is materialized and evaluated
once within the selection domain.

The cost covers preparation, packing, addresses, loads/stores, decoding,
conversion, arithmetic, reduction and explicit spill/reload. Static counts use
known extents; dynamic work retains the corresponding symbolic quantity.
Final numerical, effect and resource legality determines admissibility.

## Source improvements and compiler improvements

A better source block size or persistent layout is an author-program improvement.
A better register placement or shared supply for the same program is a compiler
improvement. For a DSL change, compare old compiler/old DSL, new compiler/old DSL,
then new compiler/new DSL so each contribution remains clear.

Use [Bench reproduction](../experiments/reproduce.md) for matching inputs and timing.
The [complete optimization contract](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/optimization-principles.md)
defines candidate boundaries and resource feedback.
