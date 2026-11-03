# Weft 1.0 release

Weft **1.0.0** is the programming model, compiler and CPU runtime release.
It extends upstream Triton **3.8.0**.

| Interface | Version |
|---|---|
| Python distribution | `triton==3.8.0+weft.1.0.0` |
| Weft API | `triton.__weft_version__ == "1.0.0"` |
| Upstream API base | `triton.__version__ == "3.8.0"` |
| Native CMake package | `find_package(Weft 1.0 CONFIG REQUIRED)` |
| Comparison project | `triton-weft-bench==1.0.0` |

## Source and packages

The public source contains the complete Triton fork, shared and target
compilation, runtime/SDK, installed `triton.kernels.weft` library, examples,
tests, bilingual handbook, licenses and sixteen result tables.
The separate [Bench project](https://github.com/Tianchen-ai/triton-weft-bench)
owns reference adapters, workload catalogs and measurement.

[Download Weft 1.0](https://github.com/Tianchen-ai/triton-weft/releases/tag/v1.0.0)
for wheels, source packages, the offline handbook and SHA256 checksums.
[Bench 1.0](https://github.com/Tianchen-ai/triton-weft-bench/releases/tag/v1.0.0)
provides its own wheel, sources and complete paper evidence archive.

Build the source distribution with `python -m build --sdist --no-isolation`.
Extract it and follow [installation](../getting-started/installation.md);
no Git history or second checkout is required. The pinned LLVM/MLIR SDK is an
external build dependency built by `scripts/build-llvm-project.sh`.
The wheel bundles Clang resources and the native runtime/SDK.

## Verify an installation

```sh
python -c 'import triton; print(triton.__weft_version__, triton.__version__)'
python -m examples.weft.quickstart --dump-ir /tmp/weft-ir --export /tmp/weft-native
python -m examples.weft.kernels.projection
python -m examples.weft.kernels.attention
python -m examples.weft.kernels.welford
python -m examples.weft.kernels.gemm
python -m pytest python/test/weft -s --tb=short
```

Examples construct inputs and check complete outputs.
The [native SDK guide](../compiler/native-sdk.md) adds C/C++ calls through the
exported ABI. [Targets and engines](../getting-started/hardware.md) explains
core selection and matrix instructions.

## Handbook and evidence

```sh
python -m pip install -r third_party/weft/requirements-docs.txt
mkdocs build --strict -f mkdocs.yml
```

This builds both handbook languages. The [result guide](../experiments/results.md)
connects approved figures, unrounded CSVs and provenance. New measurements use
the installed product with Bench; the release preserves the paper's frozen
workloads and measurements.
