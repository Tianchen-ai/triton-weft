# Paper results

These tables accompany the final manuscript. Start with the
[selected product results](https://tianchen-ai.github.io/triton-weft/experiments/results/)
or the **[complete Bench gallery](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/paper/gallery.md)**
to follow performance, author choices and compilation mechanisms. Each table
states its workload, comparison and measurement fields; ratios use matching
unrounded times. [Provenance](provenance.json) records source revisions and hashes.

| Table | Content |
|---|---|
| `operators.csv`, `operator-families.csv` | Operator cases and family summaries |
| `applications.csv` | Complete application prefill and decode |
| `application-roles.csv`, `application-x86-windows.csv` | Application role attribution and measurement windows |
| `author-controls.csv`, `author-organization.csv` | Author control and computation organization |
| `format-addition.csv` | Author source changes for adding a format |
| `increment-costs.csv`, `increment-definitions.csv` | New format/operator SLOC and individual source definitions for Figure 17 |
| `compiler-mechanisms.csv` | Compiler mechanisms under fixed programs |
| `state-algorithms.csv` | Online attention and streaming statistics |
| `ggml-compute.csv`, `grid.csv`, `fp32-libraries.csv` | Integrated compute, grids and FP32 library comparisons |

The normal kernel library is in `python/triton/kernels/weft`. Complete examples
and correctness tests are in `examples/weft/kernels` and `python/test/weft`.
