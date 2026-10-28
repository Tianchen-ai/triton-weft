# Your first CPU kernel

After [installation](installation.md), save the following file as `affine.py`.
The application owns the arrays; the kernel computes `Y = scale * X + bias`.

```python
import numpy as np
import triton
import triton.language as tl

@triton.jit(model="weft")
def affine(X, Y, N, scale, bias, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    active = offsets < N
    x = tl.load(X + offsets, mask=active, other=0.0)
    tl.store(Y + offsets, x * scale + bias, mask=active)

x = np.arange(131, dtype=np.float32)
y = np.empty_like(x)
grid = (triton.cdiv(x.size, 64),)
compiled = affine[grid](x, y, x.size, 1.25, -0.5, B=64, num_cpu_threads=2)
print(y[:4])
print("Compilation stages:", list(compiled.asm))
```

Run `python affine.py`. The first four values are
`[-0.5, 0.75, 2.0, 3.25]`.

## Understand the launch

`B` is a compile-time block size. `N`, `scale` and `bias` are runtime values.
Each `program_id` selects 64 logical elements. The three-instance grid covers
131 elements; its final mask controls actual memory access.
`num_cpu_threads=2` chooses the OpenMP execution team independently of grid size.

Call the same kernel again to reuse the compiled specialization. Changing a
constexpr such as `B` selects another specialization. Changing the input array
contents reuses the program.

## Inspect and export

```python
print(compiled.asm["weftir"])
compiled.export("/tmp/affine-artifact")  # choose a new directory
loaded = triton.load("/tmp/affine-artifact")
loaded[grid](x, y, x.size, 1.25, -0.5)
```

The exported artifact fixes `B=64`; its callable takes the remaining runtime
arguments. RISC-V exposes `riscvir` and `c`; x86 exposes `x86ir` and `llir`.
Both produce `so`.

Continue with [examples](examples.md) for row softmax and matrix multiplication,
or [Encoding](../dsl/encoding-and-artifacts.md) for packed storage.
