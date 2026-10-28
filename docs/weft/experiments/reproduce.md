# Reproduce the kernel and model experiments

Use **Triton–Weft to compile the author programs** and **Triton–Weft Bench to prepare,
check and measure the workloads**. Bench contains the installed kernel corpus,
target catalogs, reference-library recipes and model protocols.

## 1. Install and configure

Install Triton–Weft on each execution machine, then install Bench:

```sh
git clone https://github.com/Tianchen-ai/triton-weft-bench.git
cd triton-weft-bench
python -m pip install '.[verification]'
mkdir -p .local
cp configs/example.json .local/bench.local.json
export WEFT_BENCH_CONFIG="$PWD/.local/bench.local.json"
```

Edit the configuration with your target Python, allowed CPU set, native SDK,
reference-library paths and output directory. Select a catalog such as
`x86-1t`, `sg2044`, `k1` or `k3-a100`.
For SSH targets use the connection/deployment fields in the
[Bench target guide](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/targets.md).
Build references using their
[registered recipes](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/src/weft_bench/catalogs/references.json).

Target-side JIT uses `weft_bench.adapters.triton_jit:TritonJitAdapter`
with the installed target Python. Core selection and K3 registration precede
capability discovery. Local x86 follows the same workflow.

## 2. Inspect the selected program

```sh
weft-bench list --kind targets
weft-bench list --kind suites --target k1
weft-bench list --kind baselines
weft-bench show q4_k_dot --target k1
```

`show` resolves the DSL file/symbol, source parameters, target, physical options,
reference and timing scope. Base Encodings are in `triton.integrations.ggml_formats`;
explicit candidate layouts are in `weft_bench.kernels.ggml.layouts`.

## 3. Run a matched operator comparison

With a configured reference, run the selected candidate:

```sh
weft-bench op q4_k_dot --target k1 --output output/current/k1/q4_k_dot
```

To establish and reuse a GGML reference for this case:

```sh
weft-bench op q4_k_dot --target k1 --impl ggml \
  --output output/baselines/k1/q4_k_dot
weft-bench op q4_k_dot --target k1 \
  --baseline output/baselines/k1/q4_k_dot \
  --output output/current/k1/q4_k_dot
```

Bench supplies the same inputs and oracle to both implementations.
The JIT adapter executes `kernel[grid](...)` on the target and records first-call,
cache-hit and warm-call behavior. It measures that successful JIT result through
its exported native entry for the matching GGML comparison.

## 4. Run complete registered suites

After configuring the machines and their matched references:

```sh
weft-bench op --suite full --target sg2044  --output output/current/sg2044/full
weft-bench op --suite full --target k1      --output output/current/k1/full
weft-bench op --suite full --target k3-a100 --output output/current/k3-a100/full
weft-bench op --suite full --target x86-1t  --output output/current/x86-1t/full
```

| Catalog | Operator–platform cases |
|---|---:|
| SG2044 | 133 |
| K1 | 137 |
| K3 A100 | 136 |
| x86 single thread | 147 |
| Total | 553 |

Each case keeps its selected shape, author program and configuration.
For daily development, select the affected case.

## 5. Prepare a model package

Configure normal/integrated llama.cpp binaries and your GGUF file. Inspect the
model protocol and package recipe:

```sh
weft-bench list --kind models --target k1
weft-bench list --kind replacements
weft-bench prepare-model --target k1 --model llama2-7b-q4_0 \
  --output output/current/k1/llama2-package
```

The preparation result reports the deployable package path. Set the selected
model configuration's `implementations.weft.package` to that execution-side path.
The package fixes role-to-kernel bindings before inference.
The [llama.cpp tutorial](../integrations/llama-cpp.md) explains the native adapter.

## 6. Measure prefill and decode

```sh
weft-bench e2e llama2-7b-q4_0 --target k1 --impl normal --phase both \
  --output output/baselines/k1/llama2
weft-bench e2e llama2-7b-q4_0 --target k1 --impl weft --phase both \
  --baseline output/baselines/k1/llama2 \
  --validate --validation-reference output/baselines/k1/llama2-logits \
  --output output/current/k1/llama2
```

Use the catalog's input lengths, threads and repetitions. `--phase` selects the
measurement windows. In-window counters identify the bound functions that ran;
the registered logits check runs outside timing.

## Read the results

Each run writes `results.json` and tabular output with numerical status, timing,
reference, configuration and replay information. Native, grid, Python and model
calls use their respective [timing boundaries](../development/benchmarking.md).
Keep current runs in `output/current`, fixed references in `output/baselines`.

[Paper results](results.md) provides the figures, CSV tables and measured setup.
