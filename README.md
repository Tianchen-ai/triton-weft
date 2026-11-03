# Triton–Weft 1.0

[中文](README.zh-CN.md) · [Handbook](https://tianchen-ai.github.io/triton-weft/) · [Examples](examples/weft/README.md) · [Bench](https://github.com/Tianchen-ai/triton-weft-bench) · [Release guide](docs/weft/development/release.md)

**Compose high-performance CPU kernels from encoded fields, typed computation,
shared values and state.**

Weft is a programming model implemented by extending Triton. Its source objects
keep data representation, numerical groups, sharing and state updates explicit.
The compiler uses those relationships to construct vector and matrix execution
for x86 and RISC-V. Programs use `@triton.jit(model="weft")`, the normal
`kernel[grid](...)` interface, configuration, autotuning, caching and native export.

The installable library `triton.kernels.weft` supplies kernels, stored formats,
layouts, numerical helpers, tuning resources and host composition. The library
includes dense and quantized projections, normalization, online attention,
streaming statistics and stateful model components. NumPy and CPU Torch are
optional host-buffer integrations.

Weft's release is **1.0.0**, based on upstream Triton **3.8.0**. The distribution
is **`triton==3.8.0+weft.1.0.0`**. `triton.__weft_version__` identifies Weft;
`triton.__version__` preserves the upstream base identity.

## Build and install

The source tree builds without a `.git` directory. Use Python 3.10 or newer,
Git, CMake, Ninja, GCC 12 or newer and the LLVM/MLIR revision in
[`cmake/llvm-info.json`](cmake/llvm-info.json). The pinned revision supplies
Clang 23; its OpenMP runtime must use the same dispatch ABI.

To build the dependency SDK from the public LLVM project:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r python/requirements.txt numpy pytest
LLVM_PROJECT_PATH="$PWD/.deps/llvm-project" \
LLVM_BUILD_PATH="$PWD/.deps/llvm-build" \
LLVM_INSTALL_PATH="$PWD/.deps/llvm" \
bash scripts/build-llvm-project.sh
```

An existing SDK built from that exact revision can be selected instead. The
installer discovers the SDK's Clang and OpenMP paths and uses host GCC to build
the extension:

```sh
LLVM_SYSPATH="$PWD/.deps/llvm" bash scripts/build-weft.sh
python -c 'import triton; print("Weft", triton.__weft_version__, "on Triton", triton.__version__)'
```

LLVM installations with a target-specific library directory should pass that
directory's `libomp.so`. The build checks the OpenMP entry points, headers and
native Clang target. GCC compiles the product; the bundled Clang 23 compiles
generated CPU kernels. The [installation guide](docs/weft/getting-started/installation.md)
also describes explicit and cross-compilation toolchains.

## Write and call a kernel

Save this complete program as `quickstart.py` and run it after installation:

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
np.testing.assert_allclose(y, np.sum(x.astype(np.float64)**2, axis=1), rtol=1e-5)
print(y, list(compiled.asm))
```

The grid describes independent programs; `num_cpu_threads` selects the CPU
worker count. Typed `tl.View` arguments additionally expose logical fields,
shapes and storage to the compiler.

The row owns one staged input observation and one accumulator. Its inner blocks
update that same state, including the tail. The
[runnable version](examples/weft/quickstart.py) also writes the actual IR and exports
a native artifact:

```sh
python -m examples.weft.quickstart --dump-ir /tmp/weft-ir --export /tmp/weft-native
```

| Author object | Program expression | Information used by compilation |
|---|---|---|
| Stored fields | `@triton.encoding`, `tl.View` | Coordinates, bits, storage and alignment |
| Logical regions | `tl.level` | Independent outputs, grouping and active tails |
| Shared values and state | `tl.stage`, `tl.state` | Input observations, initialization and updates |
| Numerical groups | `tl.mac_groups`, `tl.widen`, `tl.reduce` | Local widths and conversion positions |
| Execution class | `tl.dot(..., engine=...)` | Vector or legal matrix construction |

Ordinary Python factories can bind Encodings and JIT helpers once. In the
library's [`organized_projection.py`](python/triton/kernels/weft/ggml/blocks/mul_mat/organized_projection.py),
Q4_K and Q6_K share the typed entry, preparation framework, weighted computation
and both output-state traversals. Each format supplies its field helper; Q4_K
also supplies its correction fields.

[![Shared fields, weighted partials and output state](docs/weft/assets/paper/model.svg)](docs/weft/assets/paper/model.svg)

[Online attention](python/triton/kernels/weft/attention/online_f32.py) and
[Welford](python/triton/kernels/weft/programs/statistics.py) compose different
coupled states through the same programming model.

## Run complete examples

Run these from the source root after installation:

```sh
python -m examples.weft.kernels.projection
python -m examples.weft.kernels.welford
python -m examples.weft.kernels.attention
python -m examples.weft.kernels.gemm
python -m pytest python/test/weft -s --tb=short
```

Every example prepares its own inputs, invokes the installed product and checks
the complete output against an independent reference. The product tests also
cover factory specialization, native export/reload, logical tails and the x86
integer/BF16 matrix paths. Matrix tests require the corresponding CPU facilities.

Host programs under `triton.kernels.weft.programs` retain the normal `Program`,
`Operators`, workspace, KV-cache, MoE and state-space interfaces. Import only the
task or host integration needed by the application. The
[handbook](docs/weft/index.md) and [example guide](examples/weft/README.md)
explain the language and deployment interfaces.

## Native deployment

Call `compiled.export(directory)` to save the native library, typed ABI and
runtime dependencies. `triton.runtime.weft.artifact.load(directory)` reloads it.
The SDK installs `<weft/runtime.h>` and `<weft/runtime.hpp>`; their source lives
in `third_party/weft/runtime/include/weft`. The native ABI and include names are
the same on both CPU families. See the
[C++ deployment example](examples/weft/deployment) and
[native SDK guide](docs/weft/compiler/native-sdk.md).

## Recorded results

The final paper dataset covers **553 operator/target cases**, **543 matched
comparisons**, and complete inference phases across eight model/platform
deployments. SG2044 inference uses 64 threads; K1, K3 and x86 use eight.
K3 CPU-vector and vendor IME2 reference policies are identified separately.

[![All matched operator comparisons](docs/weft/assets/paper/operators-ecdf.svg)](docs/weft/assets/paper/operators-ecdf.svg)

All fourteen [result tables](results/README.md), current figures and their
provenance accompany the project. The
[reproduction guide](docs/weft/experiments/reproduce.md) connects installed kernels
with Bench's input generation, strong reference dispatch, numerical checks and
native/application timing. The source package includes the complete compiler,
normal kernel library, examples and tests.

## Source map

[![Canonical program and x86/RISC-V construction](docs/weft/assets/paper/compiler.svg)](docs/weft/assets/paper/compiler.svg)

| Component | Location |
|---|---|
| Language and frontend | `python/triton/language`, `python/triton/compiler` |
| JIT, CPU backend and native calls | `python/triton/runtime`, `python/triton/backends/weft` |
| Normal CPU kernel library | `python/triton/kernels/weft` |
| Canonical IR and common analysis | `third_party/weft/lib/Dialect/Kernel`, `third_party/weft/lib/Analysis` |
| Physical RISC-V IR | `third_party/weft/lib/Dialect/RISCV` |
| Lowering, optimization and emission | `third_party/weft/lib/Target` |
| Native runtime and public SDK | `third_party/weft/lib/Runtime`, `third_party/weft/runtime/include` |
| Examples and product regressions | `examples/weft`, `python/test/weft` |

This complete fork is based on Triton v3.8.0. Upstream Triton retains its MIT
license; Weft additions retain Apache-2.0. Component attribution and third-party
notices are preserved in `LICENSE`, `NOTICE` and `third_party/weft/licenses`.
