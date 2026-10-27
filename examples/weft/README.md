# Triton–Weft examples

Install this fork's `triton` wheel and NumPy, then run from the repository root.
These examples use the installed compiler. They do not need Bench or a Python
source overlay.

| Start here | Program | Run |
|---|---|---|
| Standard Triton | [language/triton_basics.py](language/triton_basics.py) | `python -m examples.weft.language.triton_basics --operation softmax` |
| View and dynamic tails | [language/views.py](language/views.py) | `python -m examples.weft.language.views` |
| Declare packed bytes | [language/encoding.py](language/encoding.py) | `python -m examples.weft.language.encoding` |
| Level, stage and state | [language/levels.py](language/levels.py) | `python -m examples.weft.language.levels` |
| Vector or matrix contraction | [language/contraction.py](language/contraction.py) | `python -m examples.weft.language.contraction --engine vector` |
| Tune, export and load | [deployment/autotune_export.py](deployment/autotune_export.py) | `python -m examples.weft.deployment.autotune_export /path/to/new-artifact` |
| CPU PyTorch | [frameworks/pytorch_cpu.py](frameworks/pytorch_cpu.py) | `python -m examples.weft.frameworks.pytorch_cpu` |
| GGML native deployment | [deployment/llama_cpp.py](deployment/llama_cpp.py) | `python -m examples.weft.deployment.llama_cpp /path/to/new-package --profile native` |

`language/encoding.py` defines its own `@triton.encoding` beside the computation
and explicitly constructs the bytes on the host. The llama.cpp example instead
imports the shared `Q8_0` declaration from
[`triton.integrations.ggml_formats`](../../python/triton/integrations/ggml_formats.py).
That module defines storage contracts; it does not call GGML kernels.

`language/levels.py` shows the author-owned lifetime: a row's staged value and
accumulator remain available across inner block updates, including a partial
last block. `engine="matrix"` requires supported integer/BF16 hardware; use the
[target guide](../../docs/weft/compiler/runtime.md) for explicit CPU selection.

The PyTorch example requires the supported framework seam. The llama.cpp example
exports a native package; complete adapter build and worker binding steps are in
the [deployment tutorial](../../docs/weft/integrations/llama-cpp.md).
Numerical checks, reference comparisons and model timings remain in Bench.
