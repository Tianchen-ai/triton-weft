# Paper results and figure sources

These figures use the paper's measured kernel and model dataset.
The [reproduction workflow](reproduce.md) runs the corresponding Bench workloads.

## Native kernel performance

[![Kernel heatmap and full speedup distributions](../assets/paper/operators.svg)](../assets/paper/operators.svg)

| Platform | Cases | Matched comparisons | Geometric mean |
|---|---:|---:|---:|
| SG2044, RVV128 | 133 | 132 | 1.60× |
| K1 / X60, RVV256 and IME1 | 137 | 136 | 1.55× |
| K3 / A100, RVV1024 and IME2 | 136 | 135 | 1.82× |
| Xeon Platinum 8470Q, AVX-512/VNNI/AMX | 147 | 140 | 1.56× |

The distribution includes all 543 matched cases: 541 GGML references and two
oneDNN references for x86 BF16 and INT8 dense multiplication. Speedup is the
reference median divided by the candidate median for the same native workload.
Complete projections include activation quantization; packed leaves use their
declared packed-call boundary.

[Operator CSV: all 553 cases](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/paper/operators.csv)
records shape, numerical check, reference, timings and JIT observations.

## Complete inference

[![Prefill and decode for all nine paired configurations](../assets/paper/models.svg)](../assets/paper/models.svg)

The configurations cover Llama 2 7B Q4_0, DeepSeek 8B Q4_K_M and Qwen 2.5
14B Q4_K_M. All use eight application threads. K3 DeepSeek has separate
CPU–RVV and vendor IME2 reference policies.

| Protocol | SG2044 / K1 / x86 | K3 |
|---|---|---|
| Prefill | 128 tokens, initial context 0 | 32 tokens, initial context 0 |
| Decode | 32 tokens after a 512-token context | 8 tokens after a 32-token context |
| Execution | GGML workers call native package entries | GGML/vendor workers call native package entries |

[Model CSV: all 18 phases](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/paper/models.csv)
contains timing, normal reference, policy, numerical status and replacement counts.
Each plotted configuration uses its own reference policy.

## Dataset and figures

The illustrations come from the completed Weft paper V10:

| Handbook asset | Paper source | Purpose |
|---|---|---|
| `assets/paper/model.svg` | Figure 4 | Logical scopes, shared panel and partial widths |
| `assets/paper/compiler.svg` | Figure 5 | Canonical IR and two physical backends |
| `assets/paper/operators.svg` | Figure 8 | Representative roles and all matched kernel results |
| `assets/paper/models.svg` | Figure 10 | Nine paired model configurations |

The source tables are V10's `data/operators.csv` and `data/models.csv`.
Public Bench copies retain the measured values and settings.
The campaign combines target-installed JIT coverage with affected-case repairs;
case records retain the selected observations.
V10's product/Bench source snapshot is `369c22929` / `e17240a`.

[Dataset notes](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/paper/README.md)
describe the columns and reference boundaries.
