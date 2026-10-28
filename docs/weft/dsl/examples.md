# Read the examples as programs

The [runnable sequence](../getting-started/examples.md) introduces ordinary Triton,
storage, reuse and deployment. This chapter connects the source lines to their
meaning and suggests useful edits.

## Encoding: two representations of the same values

In `language/encoding.py`, the host packs 32 logical U4 elements into 16 bytes.
The declaration tells the compiler where each field element lives. The kernel
loads a record, reads `record.q` and widens it to U8 for output.

Follow one coordinate: logical element 20 reads the high nibble of byte 4.
Changing the values changes input data; changing `bit_layers` changes the storage
contract and the matching host preparation. Changing an arithmetic expression
changes what the program computes from those same fields.

## Level: a row shared by block updates

In `language/levels.py`, a row owns one input snapshot and one accumulator.
The inner block Level reads the relevant slice and contributes its squared sum.
The final partial block uses the same logical coverage and state.

Change the source parameter `B` to change the logical grouping. Then inspect
`weftir` and the target IR to see how the compiler implements that chosen grouping.
`tl.stage` marks the shared value; its physical residence is a later decision.

## Contraction: types and engine

In `language/contraction.py`, standard pointer arithmetic creates the two I8
matrix blocks. `tl.dot` produces I32 values. `ENGINE` is a constexpr passed to
the contraction. Compare `vector` and `auto`, then use `matrix` on a target
with a legal integer matrix route. Read the printed actual extensions.

## From tutorial to optimized corpus

The paper's projection combines these ideas at larger scale: encoded weight
fields, explicit partial widths, a shared panel and nested output states.
Its author programs are in the installable Bench package.

[![Source scopes and partial arithmetic in projection](../assets/paper/model.png)](../assets/paper/model.png)

[Model programs](programs.md) gives the corpus map.
[Physical optimization](../compiler/optimization-principles.md) explains how to
distinguish a useful DSL change from a compiler improvement.
