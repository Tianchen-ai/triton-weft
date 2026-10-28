# Operations, numerical widths and helpers

Standard Triton operations retain their language entry points. Weft's additions
make logical axes and numerical realization explicit when needed.

## Familiar tensor operations

| Operation | Use |
|---|---|
| `tl.arange`, indexing and broadcasting | Construct logical tiles and pointer offsets |
| `tl.load/store(..., mask=...)` | Access the active memory elements |
| `tl.sum/max`, `tl.reduce` | Reduce tensor axes; tuple combiners carry several results |
| `tl.dot(a, b, acc)` | Multiply `[..., M, K]` by `[..., K, N]`, then add the accumulator |
| `tl.cumsum`, `tl.associative_scan` | Express prefix computations |
| `tl.atomic_add` and pointer atomics | Update shared storage through an explicit atomic operation |
| `to/cast/bitcast` | Use Triton's typed conversion rules |

The [basic examples](../getting-started/examples.md) show masked tails, strided
rows, dot accumulation, scan and histogram updates.

## Explicit finite-width arithmetic

```python
partials = tl.mac_groups(weights, activation, n=4, into=tl.i16)
wide = tl.widen(partials, tl.i32)
total = tl.reduce(wide, axis="k", op="add")
```

The grouping and I16 partials are part of the numerical program. Widening occurs
before the final reduction. The compiler proves which machine instructions can
implement those exact boundaries using shared integer-range analysis.

`tl.narrow` keeps an explicit conversion boundary and its rounding/saturation
contract. Integer and packed-field results follow their exact contracts;
floating computations follow the declared rounding and error tolerance.
The compiler can fuse a local, single-consumer floating multiply/add within the
approved FMA contract.

## Named-axis contraction

```python
dot = tl.reduce_dot(w.q, x.q, over="k", acc_dtype=tl.i32, engine="vector")
```

Named-axis values align through axis identity. The reduction removes the selected
axis and preserves the others. For standard matrix tiles, use ordinary
`tl.dot(a, b, acc)`; its positional matrix convention remains available.
Both forms accept the per-contraction [engine constraint](../getting-started/hardware.md).

## Functions and source parameters

An ordinary `@triton.jit` helper is compiled with its caller's model unless it
explicitly selects a model. Helpers compose typed Values, Encoding fields,
Level scopes and effects through the common frontend.

`tl.constexpr` and declared finite source parameters choose an author program:
block extents, group sizes and layout instances. `triton.Config.kwargs` binds
those parameters. `backend_options` controls physical choices within that program.
The [autotune tutorial](../compiler/runtime.md) shows the two in one workflow.

See the [operation contract](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/dsl/functions-and-operations.md)
for signedness, NaN/signed-zero behavior, reduction identities, widening,
narrowing and helper semantics.
