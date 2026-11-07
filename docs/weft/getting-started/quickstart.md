# Your first Weft kernel

This complete row computation combines typed storage, shared input and ordered
state updates. After [installation](installation.md), run
`python -m examples.weft.quickstart`.

## Write and call the program

Save this as `row_sum_squares.py` and run `python row_sum_squares.py`:

```python
from __future__ import annotations
import numpy as np
import triton
import triton.language as tl

@triton.jit(model="weft")
def row_sum_squares(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M,)],
                    BLOCK: tl.constexpr = 32):
    with tl.level.rows(M, group=1) as row:
        values = tl.stage(tl.load(X[row]))
        total = tl.state(tl.f32, [1], init=0)
        with tl.level.blocks(N, extent=BLOCK) as block:
            part = values[:, block]
            total += tl.reduce(part * part, axis="n")
        tl.store(Y[row], total)

x = np.random.default_rng(19).normal(size=(4, 65)).astype(np.float32)
y = np.empty(4, dtype=np.float32)
compiled = row_sum_squares[(1,)](x, y, BLOCK=32, num_cpu_threads=1)
np.testing.assert_allclose(
    y, np.sum(x.astype(np.float64) ** 2, axis=1), rtol=1e-5, atol=1e-5)
print(y, list(compiled.asm))
```

Every output is checked against a double-precision reference.

## Follow the source relationships

View types bind `M` and `N` from the input shape. A row owns one staged
input observation and one initialized state. Inner blocks contribute 32, 32
and one active element to that same state. The final store observes the complete
update chain. Changing `BLOCK` specializes the logical grouping.

The compiler constructs target vectors and local storage from these relationships.
`num_cpu_threads` selects the CPU worker count; the grid describes independent
program instances. Calls with the same types and bindings reuse compilation.

## Inspect and export

The supplied program also saves actual compiler IR and exports a native artifact:

```sh
python -m examples.weft.quickstart \
  --dump-ir /tmp/weft-quickstart-ir --export /tmp/weft-quickstart-native
```

`compiled.asm` provides Canonical `weftir`, target `x86ir` or `riscvir`,
and `llir` or `c`. Export saves the native library, ABI and dependencies.
The [native SDK chapter](../compiler/native-sdk.md) shows loading and C/C++ calls.

Continue with [Encoding](../dsl/encoding-and-artifacts.md),
[Level and state](../dsl/values-and-levels.md), or complete
[projection, attention, Welford and GEMM examples](examples.md).
