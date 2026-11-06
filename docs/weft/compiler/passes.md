# Passes, selection and software pipelines

Passes transform explicit programs. Each pass consumes established facts, records
its physical decisions and updates analyses affected by rewriting.

## The recurring pipeline structure

| Stage | Main work |
|---|---|
| Verify source contracts | Types, axes, effects, Encoding and Level initialization |
| Import into target representation | Carry logical identity and map storage access |
| Choose layouts and supply | Compare finite legal representations and shared consumers |
| Materialize | Insert actual loads, packing, conversions and local target operations |
| Schedule | Place operations subject to SSA, memory and lifetime dependencies |
| Close resources | Analyze live representations, local storage and spill/reload |
| Verify and emit | Check selected operations, then translate |

Candidate evaluation and final compilation use the same materialization machinery.
Cost and resource analysis therefore inspect the actual physical work, including
preparation and other consumers.

## Shared analyses

`IntegerFacts` supplies range and signedness proofs, including statically bounded
ordinary loop induction values. Carries retain their own numerical facts.
`LogicalAxes` and
`IndexRelations` retain coordinate identity. `StorageBytes`,
`StorageLifetime` and `PhysicalDependence` describe payload accesses, snapshots,
aliasing and birth/handoff constraints. These analyses live above target-specific
instruction selection.

SCF simplification can create scalar `arith.select`; control normalization maps
it back to Canonical select with the same operands and type. Streamed producer
graphs enumerate target-legal widths independently of their initial packet,
charging each candidate's temporaries, live values and reduction carries.

Bounded loop coordinates can discard an index/integer/index round trip only
when the shared range proof fits the intermediate carrier. RISC-V memory
selection also proves that scaled lookup offsets fit their selected unsigned
carrier, inserting an explicit extension when necessary. Saved table lookups
carry index and table validity without moving their read across an aliasing write.

## Software pipeline depth

```python
configs = [
    triton.Config({"B": 64}, backend_options={"pipeline_depth": depth})
    for depth in (1, 2, 3, 4)
]
```

Depth 1 is the default. A scheduler selects a legal stage/order for the current
program; the shared expander constructs prologue, steady state, epilogue,
cross-stage SSA and private-buffer versions.

The expanded program is rechecked for valid domains, dependencies and resources.
Buffer versions follow the overlap between production and last use.
Source `tl.stage` continues to define semantic ownership; physical pipeline depth
defines execution scheduling.

## Finding an implementation

RISC-V and x86 assemble their pass pipelines in their `Backend` directories.
Target transforms are grouped by arithmetic, contraction, memory, control and
resource responsibilities. Shared full-tile and software-pipeline transforms live
in `third_party/weft/lib/Transforms`.

The [RISC-V pass specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/passes.md)
and [x86 specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/x86.md)
provide operation-level preconditions, transformations and verification rules.
