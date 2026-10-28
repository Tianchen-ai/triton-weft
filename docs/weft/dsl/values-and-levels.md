# Values, computation levels and state

A Value has an element type, logical shape and axis identities. A Level gives a
group of computations its domain, partition and lifetime. Together they make
reuse and accumulation part of the author program.

## The four operations

| Form | Meaning |
|---|---|
| `tl.load(region)` | Read a Value at this point in the ordered program |
| `tl.stage(expression)` | Create a read-only value owned by this scope for descendants to reuse |
| `tl.state(dtype, shape, init=...)` | Initialize updateable state owned by this scope |
| `tl.store(region, value)` | Write the selected logical region |

Move a state declaration to change when it initializes. Move a stage declaration
to change which descendants share it. These are explicit source-level choices.

## Select a domain

```python
with tl.level.rows(M, group=1) as row:
    values = tl.stage(tl.load(X[row]))
    total = tl.state(tl.f32, [1], init=0)
    with tl.level.blocks(N, extent=32) as block:
        part = values[:, block]
        total += tl.reduce(part * part, axis="n", op="add")
    tl.store(Y[row], total)
```

For a 65-element row the inner Level covers 32, 32 and 1 valid elements.
The row owns the accumulated result across all three instances.
`rows/cols(..., group=...)` express logical cohorts; `tiles/blocks(..., extent=...)`
partition an axis; `subtiles(parent, extent=...)` refine a parent region.

The target may execute a cohort in several physical issues. Its logical ownership
and initialization count remain those expressed by the author.

## Initialization and visibility

At a Level entry, state initialization precedes stage initialization, then the
body runs. Each initializer uses values visible before entry, the current Level
point, and earlier values in its own initialization region.

Write the staged input directly:

```python
with tl.level.rows(M, group=1) as row:
    values = tl.stage(tl.load(X[row]))
    # Descendants consume values.
```

This makes the read's scope explicit. A body-local temporary belongs to the body;
an initializer captures the incoming SSA environment.

## Updates and handoff

Ordinary assignments express state changes: `total += contribution`,
`m = new_m`, or `o = o * alpha + contribution`.
Canonical IR carries all updated values through the Level result. Several carries,
ordinary branches and nested loops use the same SSA relationships.

Ordinary `for`, `while` and `if` express ordered scalar control. Use them for
index traversal or dependent search; use shaped Values when an algorithm needs
an explicit logical axis.

## Source stage and software pipeline

`tl.stage` specifies a value's ownership and reuse.
`backend_options={"pipeline_depth": 3}` requests a physical software schedule of
that program. The scheduler checks dependencies; the expander versions cross-stage
values and private buffers. The [pipeline chapter](../compiler/passes.md) explains the distinction.

[Complete Value and Level semantics](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/dsl/values-and-levels.md).
