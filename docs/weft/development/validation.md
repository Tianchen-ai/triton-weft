# Numerical checks

Use the registered workload's numerical contract when running an example,
changing a DSL program or improving a compiler pass.

| Data or operation | Check |
|---|---|
| Encoding and prepared packed storage | Exact field/byte mapping, padding and layout identity |
| Integer arithmetic | Declared signedness, width, wrapping and accumulator behavior |
| Floating output | Workload-specific absolute/relative tolerance and finite-value rules |
| Mutable state and aliases | Complete update and storage relationship |
| Model integration | Registered logits check and replacement counters for the measured phase |

## Use Bench's existing entry

```sh
weft-bench op q4_k_dot --target k1 --check-only \
  --output output/current/k1/q4_k_dot-check
weft-bench verify --target k1 --list
```

Select a verification ID from that list when checking a language or runtime feature:

```sh
weft-bench verify --target k1 --case 'operators/q4_k_dot' \
  --output output/current/k1/verification
```

The configured suite defines exact check coverage. Leaf and decode cases check
their complete registered outputs; large GEMM/prefill workloads may combine
sampled scalar outputs with full-output finiteness. Results retain that distinction.

## A focused repair loop

Keep the failing input, source/configuration and generated program. Repair the
responsible layer, then rerun that case and directly affected shared paths.
The [reproduction guide](../experiments/reproduce.md) covers baseline reuse,
result inspection and model phases.

[Detailed numerical and measurement contract](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/development/validation.md).
