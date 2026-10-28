# Encoding, View and persistent data

An Encoding maps a logical field coordinate to storage bytes and bits. Arithmetic
in the kernel gives those fields their numerical meaning. A View combines this
storage contract with logical shape and access information.

## Declare a packed record

The runnable [Encoding example](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/language/encoding.py)
uses 32 unsigned four-bit values in 16 bytes:

```python
@triton.encoding
class PackedNibbles:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 32
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)
```

For field coordinate `i`, the storage byte is `i % 16`.
Coordinates 0–15 use its low nibble; 16–31 use its high nibble.
`elements` describes the logical elements per record; the field mapping determines
the record's byte span.

The host prepares those bytes explicitly:

```python
logical = (np.arange(64, dtype=np.uint8) % 16).reshape(2, 32)
storage = logical[:, :16] | (logical[:, 16:] << 4)
x = tl.View(storage, shape=(logical.size,), encoding="PackedNibbles")
```

Inside the kernel, `tl.load(X[block]).q` obtains the field as a logical Value.
`tl.widen(record.q, tl.u8)` changes its numerical width while retaining coordinates.

## Choose the appropriate declaration

| Construct | Meaning |
|---|---|
| Dense `tl.View[tl.f32, (M, N)]` | Typed row-major storage with logical extents |
| `grouped(elements=N)` | Repeat a storage mapping independently in each N-element group |
| `bit_layers(elements=P, order=...)` | Place corresponding elements of P-sized layers in byte bit ranges |
| `pack_fields(group=..., fields=..., low_bits=..., order=...)` | Several adjacent fields share one compact byte container |
| `alignment` and explicit padding | Record-start alignment and storage ranges without logical fields |

For GGML formats, import the shared declarations by their purpose:

```python
from triton.integrations import ggml_formats
# Example annotation: tl.View[ggml_formats.Q4_K, (N, K)]
```

The module contains base storage declarations. Bench's
`weft_bench.kernels.ggml.layouts` contains explicit candidate layouts and builders,
organized by IME1, IME2, AMX and row arrangement.

## Views, slices and aliasing

A View slice selects a memory region; `tl.load` captures its value at that point
in the program. Dense nonnegative strides are declared with
`@triton.jit(model="weft", strided=("X",))`.
Read-only zero-stride broadcasting is supported. Writable Views must have
non-overlapping element storage.

`alias_groups={"X": "input", "Y": "output"}` declares distinct storage groups.
Arguments in one group may overlap; different groups are the caller's disjointness
promise. Omitted arguments remain in a shared may-alias group.

## Persistent layouts and local packing

`@triton.derive` describes an explicit persistent layout family.
For example, `tl.interleave(W, rows=16)` creates a byte-major cohort of 16 base
records; the resulting layout identity is shared by preparation and consumption.
Its initialization cost and storage belong to the application.

Local packing of an already-loaded Value is a compiler representation choice
within that Value's lifetime. The physical program records its allocation,
conversion, use and release. This distinction lets a source layout remain stable
while RVV, IME and AMX use different local arrangements.

The [complete Encoding and storage contract](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/dsl/encoding-and-artifacts.md)
includes field-coordinate formulas, derived layouts, padding, tails and pinned ABI boundaries.
