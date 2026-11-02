# Triton–Weft

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

## Build and install

The source tree builds without a `.git` directory. Use Python 3.10 or newer,
CMake, Ninja, a C++20 compiler and the LLVM/MLIR revision in
[`cmake/llvm-info.json`](cmake/llvm-info.json). The pinned revision supplies
Clang 23; its OpenMP runtime must use the same dispatch ABI.

To build the dependency SDK from the public LLVM project:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r python/requirements.txt
LLVM_PROJECT_PATH="$PWD/.deps/llvm-project" \
LLVM_BUILD_PATH="$PWD/.deps/llvm-build" \
LLVM_INSTALL_PATH="$PWD/.deps/llvm" \
bash scripts/build-llvm-project.sh
```

An existing SDK built from that exact revision can be selected instead. Build
the product with the SDK, host Clang and OpenMP locations explicitly:

```sh
CC="$PWD/.deps/llvm/bin/clang" \
CXX="$PWD/.deps/llvm/bin/clang++" \
LLVM_SYSPATH="$PWD/.deps/llvm" \
TRITON_WEFT_CLANG_ROOT="$PWD/.deps/llvm" \
TRITON_WEFT_OPENMP_LIBRARY="$PWD/.deps/llvm/lib/libomp.so" \
TRITON_WEFT_OPENMP_INCLUDE_DIR="$PWD/.deps/llvm/lib/clang/23/include" \
TRITON_BUILD_DIR="$PWD/.build" \
TRITON_BUILD_PROTON=OFF \
python -m pip install --no-build-isolation .
python -m pip install numpy pytest
```

LLVM installations with a target-specific library directory should pass that
directory's `libomp.so`. The build checks the OpenMP entry points, headers and
native Clang target. The [installation guide](docs/weft/getting-started/installation.md)
also describes explicit and cross-compilation toolchains.

## Write and call a kernel

```python
import numpy as np
import triton
import triton.language as tl

@triton.jit(model="weft")
def affine(X, Y, N, scale, bias, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + i, mask=i < N, other=0.0)
    tl.store(Y + i, value * scale + bias, mask=i < N)

x = np.arange(131, dtype=np.float32)
y = np.empty_like(x)
compiled = affine[(triton.cdiv(x.size, 64),)](
    x, y, x.size, 1.25, -0.5, B=64, num_cpu_threads=2)
np.testing.assert_array_equal(y, x * 1.25 - 0.5)
```

The grid describes independent programs; `num_cpu_threads` selects the CPU
worker count. Typed `tl.View` arguments additionally expose logical fields,
shapes and storage to the compiler.

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

The accompanying `results/*.csv` files contain the measured operator, model and
controlled-study results with their comparison boundaries. Values and counts
come from those files. The source package contains the complete compiler,
normal kernel library, examples and correctness tests.

## Source map

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
