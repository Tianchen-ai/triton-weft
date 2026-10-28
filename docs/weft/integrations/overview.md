# CPU tensors, PyTorch and composition

CPU tensors can be passed to the same Triton kernel used with NumPy.
The runtime borrows their storage and preserves dtype, shape, stride, offset and aliases.

## Eager Tensor calls

```python
import torch
import triton
from examples.weft.language.triton_basics import affine

x = torch.linspace(-1, 1, 131, device="cpu")
y = torch.empty_like(x)
affine[(triton.cdiv(x.numel(), 64),)](
    x, y, x.numel(), 1.25, -0.5, B=64, num_cpu_threads=2
)
```

The application allocates output and selects the grid. Read/write effects and
storage relationships come from the same Canonical interface used by compilation.

## Inductor-generated kernels

In the supported PyTorch 2.14 environment, install the source seam once:

```sh
python -m triton.tools.pytorch
python -m examples.weft.frameworks.pytorch_cpu
```

The example compiles this graph:

```python
def normalize(value):
    centered = value - value.mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(
        (centered * centered).mean(dim=-1, keepdim=True) + 1e-5
    )

compiled = torch.compile(normalize, backend="inductor", fullgraph=True,
                         options={"cpu_backend": "triton"})
```

Inductor supplies graph decomposition, fusion and Triton generation.
The generated kernel uses the shared frontend and CPU compilation path.
The seam carries model identity, effects, FakeTensor metadata and launch behavior.

## State and gradients

The host owns model state, workspace and preparation. Use explicit sequences
when several kernels cooperate. Author backward kernels and register them through
the framework's autograd interfaces; the corpus includes a
`weft_bench.kernels.dense.torch_affine` example.

Continue with [model composition](../dsl/programs.md), [native SDK](../compiler/native-sdk.md)
or [llama.cpp deployment](llama-cpp.md).
