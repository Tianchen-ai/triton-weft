# Learn Triton–Weft by running it

[中文](README.zh-CN.md) · [Handbook](https://tianchen-ai.github.io/triton-weft/getting-started/examples/)

Install the project's Triton wheel and NumPy, clone this repository, and run the
commands below **from the repository root**. Each file contains a kernel and its
host application. The application prepares data, launches the kernel and prints
the result.

## 1. Familiar Triton

```sh
python -m examples.weft.language.triton_basics --operation affine
python -m examples.weft.language.triton_basics --operation softmax
python -m examples.weft.language.triton_basics --operation matmul
python -m examples.weft.language.triton_basics --operation scan
python -m examples.weft.language.triton_basics --operation histogram --threads 2
```

Open [triton_basics.py](language/triton_basics.py) beside the output:

| Operation | Input and expected result | Read this part of the kernel |
|---|---|---|
| Affine | 131 F32 values → `1.25 * x - 0.5` | Three grid instances and a masked tail |
| Softmax | Four rows of 117 F32 values → each row sums to one | Runtime row stride, max and sum reductions |
| Matmul | `[16,32] @ [32,16]` → `[16,16]` | Two-dimensional grid, K loop and dot accumulator |
| Scan | Four rows of 31 I32 values → inclusive prefix sums | `tl.cumsum` and final mask |
| Histogram | 131 indices in 16 bins → counts 9,9,9,8,… | Shared output initialization and `tl.atomic_add` |

## 2. Storage and reuse

```sh
python -m examples.weft.language.views --size 4097
python -m examples.weft.language.encoding
python -m examples.weft.language.levels
```

- [views.py](language/views.py): `N` comes from the typed View. A subview covers the
  last element of the last grid instance. Change `--size` to observe the dynamic extent.
- [encoding.py](language/encoding.py): `PackedNibbles` declares 32 U4 values in 16
  bytes. The host packs low/high nibbles; the kernel reads the logical `q` field
  and widens it to U8. The first record prints `0…15, 0…15`.
- [levels.py](language/levels.py): a row's stage holds an input snapshot and its
  state accumulates squared block contributions. The 65 columns form blocks
  of 32, 32 and 1. Follow the ownership of `values` and `total` before changing `B`.

Base GGML Encodings are declared in
[`triton.integrations.ggml_formats`](../../python/triton/integrations/ggml_formats.py).
The standalone Encoding example declares its own layout next to the consumer.
Bench's `weft_bench.kernels.ggml.layouts` contains explicit hardware-oriented layouts.

## 3. Vector and matrix arithmetic

```sh
python -m examples.weft.language.contraction --engine vector
python -m examples.weft.language.contraction --engine auto
```

[contraction.py](language/contraction.py) multiplies an I8 `[16,64]` block by an
I8 `[64,16]` block and returns I32. It prints actual artifact extensions and
physical IR stages. On a configured AMX host, choose an allowed CPU and run:

```sh
python -m triton.tools.cpu --profile x86-amx --cpus 0 \
  examples/weft/language/contraction.py --engine matrix
```

Use [hardware selection](../../docs/weft/getting-started/hardware.md) for K1/K3
registration, core groups and the available contraction routes.

## 4. Autotune, export and native C++

```sh
python -m examples.weft.deployment.autotune_export /tmp/tuned-affine
python -m examples.weft.language.views --size 64 --threads 1 --output /tmp/affine-native
python -c "from triton.runtime.package import sdk_prefix; print(sdk_prefix())"
cmake -S examples/weft/deployment -B /tmp/weft-native-build \
  -DWeft_DIR=/printed/sdk/path/lib/cmake/Weft
cmake --build /tmp/weft-native-build
/tmp/weft-native-build/weft-native-call /tmp/affine-native
```

Use new artifact directories. [autotune_export.py](deployment/autotune_export.py)
selects B from 32/64/128, exports the winner and calls the loaded artifact.
[native_call.cpp](deployment/native_call.cpp) loads the 64-element View program,
binds caller storage and prints `-0.5 0.75 2 3.25`.

## 5. Frameworks and models

| Example | Setup and next step |
|---|---|
| [pytorch_cpu.py](frameworks/pytorch_cpu.py) | Install supported CPU PyTorch and run `python -m triton.tools.pytorch`; then `python -m examples.weft.frameworks.pytorch_cpu` demonstrates eager Tensor calls and Inductor-generated normalization |
| [llama_cpp.py](deployment/llama_cpp.py) | `python -m examples.weft.deployment.llama_cpp /tmp/q8-package --profile native` exports a Q8_0 row-dot; follow the [deployment tutorial](../../docs/weft/integrations/llama-cpp.md) to build and bind GGML |

For optimized kernel programs, numerical checks and performance comparisons,
use [Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench).
The [reproduction guide](../../docs/weft/experiments/reproduce.md) connects these
language examples to the paper's operator and model workloads.
