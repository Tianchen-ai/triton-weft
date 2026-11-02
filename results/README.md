# Recorded results

These CSVs accompany the V19 manuscript. Each table states its workload,
comparison and measurement fields. Ratios use matching unrounded times.

| Table | Content |
|---|---|
| `operators.csv`, `operator-families.csv` | Operator cases and family summaries |
| `applications.csv` | Complete application prefill and decode |
| `application-roles.csv`, `application-x86-windows.csv` | Application role attribution and measurement windows |
| `author-controls.csv`, `author-organization.csv` | Author control and computation organization |
| `format-addition.csv` | Author source changes for adding a format |
| `compiler-mechanisms.csv` | Compiler mechanisms under fixed programs |
| `state-algorithms.csv` | Online attention and streaming statistics |
| `ggml-compute.csv`, `grid.csv`, `fp32-libraries.csv` | Integrated compute, grids and FP32 library comparisons |

The normal kernel library is in `python/triton/kernels/weft`. Complete examples
and correctness tests are in `examples/weft/kernels` and `python/test/weft`.
