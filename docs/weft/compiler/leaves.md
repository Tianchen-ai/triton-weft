# Local instruction contracts

A local target operation describes a closed piece of machine work: its typed
operands, coordinate coverage, numerical behavior, transfers, effects and resources.
Physical passes select it after checking those contracts.

## Contract fields

| Field | Example question |
|---|---|
| Types and widths | Can this instruction represent the source product and accumulator? |
| Axes and valid domain | Which free and reduction coordinates does it cover? |
| Operand layout | Which packet, register group or matrix fragment does it consume? |
| Numerical semantics | Does it preserve signedness, wrapping, rounding and reduction boundaries? |
| Effects and transfers | Which bytes or local storage versions does it read and write? |
| Resources and capabilities | Which registers, scratch bytes and ISA extensions are required? |

For a grouped integer operation, common integer facts prove whether a widening
or saturating machine operation is legal. For a matrix operation, physical
operand layouts and accumulator representation must agree with the selected contract.

## Extension modules

An IME or AMX operation participates in ordinary physical compilation:
capability declaration → legality → selection → materialization → verification → emission.
Outer traversal and source state remain represented by the surrounding program.

[Hardware extension guide](../development/extensions.md) ·
[Complete RISC-V local-operation specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/leaves.md).
