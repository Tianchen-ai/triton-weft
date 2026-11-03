# Final paper results and figure sources

These figures use the final paper dataset. The [reproduction workflow](reproduce.md)
connects the installed product with Triton–Weft Bench. Speedup divides reference
time by Weft time; compiler interventions divide disabled time by enabled time.

## Native operator performance

[![Representative operator roles](../assets/paper/operators-heatmap.svg)](../assets/paper/operators-heatmap.svg)

[![Distributions over every matched comparison](../assets/paper/operators-ecdf.svg)](../assets/paper/operators-ecdf.svg)

| Platform | Cases | Matched comparisons | Geometric mean |
|---|---:|---:|---:|
| SG2044, RVV128 | 133 | 132 | 1.60× |
| K1/X60, RVV256 and IME | 137 | 136 | 1.54× |
| K3/A100, RVV1024 and IME2 | 136 | 135 | 1.83× |
| Xeon Platinum 8470Q, AVX-512/VNNI/AMX | 147 | 140 | 1.32× |

The distributions include all 543 matched cases. Optimized GGML supplies the
principal references and oneDNN supplies two BF16/INT8 dense references.
Complete projections include activation preparation; microkernels consume
prepared operands. The `scope` column identifies each timed interface.

## Complete inference

[![Complete prefill and decode comparisons](../assets/paper/models.svg)](../assets/paper/models.svg)

Eight model/platform deployments cover Llama 2 7B, DeepSeek 8B and Qwen 2.5 14B.
The 18 phase comparisons include both K3 vendor IME2 and CPU-vector references.
SG2044 uses **64 threads**; K1, K3 and x86 use **eight**.

SG2044/K1/x86 measure 128-token prefill and 32-token decode after a 512-token
context. K3 measures 32-token prefill and eight-token decode after 32 tokens.
Pairs fix models, weights, shapes and threads. Application CSVs record the
numerical checks and process aggregation.

## Final tables

| Table | Rows | Evidence |
|---|---:|---|
| [operators.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/operators.csv) | 553 | All operator/target cases and matched times |
| [operator-families.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/operator-families.csv) | 5 | Family counts and geometric means |
| [applications.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/applications.csv) | 18 | Complete inference phases and reference policies |
| [application-x86-windows.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/application-x86-windows.csv) | 120 | x86 per-process stage observations |
| [application-roles.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/application-roles.csv) | 18 | Computation and preparation role effects |
| [fp32-libraries.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/fp32-libraries.csv) | 12 | FP32 GEMM against three libraries |
| [grid.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/grid.csv) | 48 | Complete calls on 1/2/4/8 threads |
| [ggml-compute.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/ggml-compute.csv) | 32 | Computation inside GGML workers |
| [author-controls.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/author-controls.csv) | 240 | Five-route sharing and precision controls |
| [author-organization.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/author-organization.csv) | 48 | Arithmetic and output-state scope |
| [five-route-bindings.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/five-route-bindings.csv) | 30 | Selected task bindings |
| [format-addition.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/format-addition.csv) | 5 | Q4 construction and Q6 source increments |
| [state-algorithms.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/state-algorithms.csv) | 36 | Attention and Welford organizations |
| [compiler-mechanisms.csv](https://github.com/Tianchen-ai/triton-weft/blob/main/results/compiler-mechanisms.csv) | 214 | Paired physical mechanisms |

Control tasks compare complete calls in intrinsic C, Triton CPU, Exo 2,
TensorIR and Weft. State studies compare sequential and blocked organizations;
compiler studies fix source semantics and change physical mechanisms.
Sample columns preserve batch observations; ratios use unrounded values.

## Diagram and dataset provenance

Current vector figures explain shared fields and state, common/target
compilation, state partition, logical/physical layouts and encoding reuse.
Their names identify their roles.
The [provenance manifest](https://github.com/Tianchen-ai/triton-weft/blob/main/results/provenance.json) records the paper source
revision, original figure paths, hashes, row counts and thread settings.
Product and Bench CSV copies match this snapshot. The
[Bench dataset guide](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/paper/README.md)
explains the columns and timing boundaries.
