# Model programs and host composition

The installable `weft_bench.kernels` package contains complete author programs.
They use the same Triton language, compiler and runtime as the tutorials.

| Program family | Computation expressed by the source |
|---|---|
| Normalization | RMSNorm, centered-variance LayerNorm, residual fusion and explicit row/tile variants |
| Activation | SiLU, SwiGLU, softplus, add, copy, fill and selection |
| Linear | Dense contraction, bias/activation epilogues and nested Level variants |
| Attention | Split/interleaved RoPE, KV writes and online causal grouped-query attention |
| MoE | Router top-k, token grouping, ragged expert projection and weighted merge |
| Mamba | Causal depthwise convolution and selective state recurrence |

## Call a program

```python
from weft_bench.kernels.programs.normalization import rms_norm
rms_norm[(1,)](X, Weight, Y, epsilon=1e-5, BLOCK=32)
```

This program has a complete row traversal: X and Y are two-dimensional Views,
Weight is one-dimensional, and BLOCK is an explicit source parameter.
Its single instance covers the source traversal.

## Compose host operations

```python
from weft_bench.kernels.programs import get, Operators

config = get("rms_norm").config(BLOCK=64)
with Operators(configs={"rms_norm": config}) as operators:
    result = operators.rms_norm(x, weight)
```

Host composition allocates outputs, checks application shapes and calls kernels
in source order. Configurations are Triton Config objects; tuning candidates use
the same Triton autotuner.

Applications own KV cache, workspace, prepared weights and recurrent state.
They decide when preparation runs and which calls share that state.
Training uses explicitly authored backward kernels or framework registration.

Use the [Bench catalog](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/catalogs.md)
to inspect source symbols, selected layouts and registered workloads.
