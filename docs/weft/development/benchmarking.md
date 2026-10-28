# References and timing boundaries

A useful comparison holds the mathematical workload, inputs, layout, preparation,
threads and timing boundary constant. Bench catalogs record these choices.

## Choose a reference

| Workload | Registered reference families |
|---|---|
| Quantized dot, dequantization, quantization, projections and CPU operations | GGML CPU/RVV or x86 implementations |
| Packed matrix calls | The matching vendor IME1/IME2 or GGML x86 matrix route |
| Dense multiplication | GGML, explicit OpenBLAS or oneDNN configuration |
| Complete inference | Normal llama.cpp with the selected CPU/vendor backend |

Inspect `weft-bench list --kind baselines` and `show CASE --target TARGET`.
The source version, build recipe and actual implementation remain part of the reference.

## Read the correct clock

| Measurement | Included work |
|---|---|
| Native workload | Its complete native call, including registered algorithmic preparation |
| Grid | All instances, OpenMP launch and synchronization |
| Warm Python JIT | Python binding, dispatch and normal launcher |
| First JIT call | Compilation, loading and initial execution |
| Model phase | Actual prefill or decode window in llama.cpp |

Speedup is reference median divided by candidate median for the same clock.
For target-side JIT, Bench also measures the exported native result against the
native reference. The warm Python and first-call measurements are recorded separately.

## Reuse and paired measurements

Reuse a saved matched reference with `--baseline`. If machine frequency or runtime
state has changed, keep the same reference library/version and compare in the same
session. Bench's `--sample-order alternating` supports configured native
implementations on shared inputs. Keep historical numbers alongside their original
settings; the current paired comparison describes the current machine state.

[Reference and attribution specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/development/benchmarking.md).
