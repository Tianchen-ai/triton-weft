# Runnable examples

The examples are short applications with visible data preparation and output.
Install Triton–Weft and NumPy, then run from the repository root.
Keep the [source index](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/README.md)
open while following the sequence below.

## 1. Ordinary Triton

```sh
python -m examples.weft.language.triton_basics --operation affine
python -m examples.weft.language.triton_basics --operation softmax
python -m examples.weft.language.triton_basics --operation matmul
python -m examples.weft.language.triton_basics --operation scan
python -m examples.weft.language.triton_basics --operation histogram --threads 2
```

| Example | What to inspect | Expected behavior |
|---|---|---|
| Affine | `program_id`, arange and tail mask | 131 outputs equal `1.25*x-0.5` |
| Softmax | Runtime row strides and max/sum | Four normalized rows of 117 elements |
| Matmul | Two-dimensional grid and K accumulation | A `16×16` output from `16×32 @ 32×16` |
| Scan | Inclusive `tl.cumsum` | Prefix sums for each 31-element I32 row |
| Histogram | Atomic updates of shared counts | 131 observations across 16 bins |

## 2. Add explicit relationships

```sh
python -m examples.weft.language.views --size 4097
python -m examples.weft.language.encoding
python -m examples.weft.language.levels
python -m examples.weft.language.contraction --engine vector
```

Views express dynamic extents and subregions. Encoding connects packed bytes to
logical fields. Levels connect a staged row, its accumulator and block updates.
Contraction prints the selected artifact's actual extensions.
The [worked-example chapter](../dsl/examples.md) explains each decision.

## 3. Tune and deploy

```sh
python -m examples.weft.deployment.autotune_export /tmp/tuned-affine
```

This tunes the source block size, exports the selected kernel, reloads it and
executes it. Continue with the complete [C++ application](../compiler/native-sdk.md)
or [llama.cpp integration](../integrations/llama-cpp.md).

## 4. CPU tensors and graph compilation

```sh
python -m triton.tools.pytorch
python -m examples.weft.frameworks.pytorch_cpu
```

With the supported PyTorch environment, the example first calls a handwritten
kernel on CPU tensors, then compiles a normalization graph through Inductor.
See [framework setup and flow](../integrations/overview.md).

Bench maintains independent numerical checks for these same applications and the
full optimized corpus. Follow [experiment reproduction](../experiments/reproduce.md)
to compare kernels and complete inference.
