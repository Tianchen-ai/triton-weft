# RISC-V Physical IR

The RISC-V physical dialect records how one Canonical program executes with RVV
and optional matrix extensions. Inspect `compiled.asm["riscvir"]` after compilation.

## Follow an encoded contraction

1. View/Encoding lowering establishes the exact storage relation and valid coordinates.
2. Layout passes assign free and reduction coordinates to lanes, issues and replicas.
3. Supply passes materialize loads, decoding, local packing and representation conversions.
4. Instruction selection chooses legal RVV or IME local operations.
5. Scheduling and resource passes close buffer lifetimes, live register groups and spills.
6. Final verification checks that emission has all required decisions.

A partial tail carries an explicit valid domain. Reduction fills inactive physical
coordinates with the operation's identity. Conversions keep the source's
signedness, width and axis identities.

## VLEN and matrix extensions

VLEN is a target fact used by layout, resource analysis and native compilation.
The same logical program can require different packet and issue counts at
VLEN 128, 256 and 1024.

IME1 and IME2 add local matrix contracts and capability checks. Their operands,
accumulators and transfer requirements are represented in the physical program.
The author's per-contraction `engine` constrains selection.

## Source locations

- `Dialect/RISCV/IR`: custom types, operations and verification.
- `Target/RISCV/Conversion`: Canonical import.
- `Target/RISCV/Transforms`: representation, arithmetic, memory, contraction, control and resources.
- `Target/RISCV/Extensions`: extension-specific instruction contracts.
- `Target/RISCV/Emission`: intrinsic C and local assembly.

[Complete physical operation specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/riscv-ir.md).
