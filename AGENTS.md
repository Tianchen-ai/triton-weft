# Working on Triton

## Weft CPU extension

- This fork uses `main` for the product, with upstream `v3.8.0` as its base. Keep upstream changes focused on the documented extension seams in `docs/weft/development/upstream.md`.
- Weft supports RISC-V intrinsic C/local assembly and x86 LLVM IR. Preserve Canonical numerical, Encoding and lifecycle contracts; physical passes own layout, local instruction selection and resource decisions.
- Authors constrain each contraction with `engine`. Default to a legal matrix implementation, otherwise vector; an explicit matrix requirement must not silently execute a vector route.
- Weft verification, optimized kernel corpus, reference libraries and performance outputs belong to the independent `triton-weft-bench` project. Use its affected cases and fixed baselines; do not add a second Weft test runner here. The upstream testing instructions below continue to apply to upstream GPU work.
- Product examples must work with the installed `triton` wheel. Compiler and generic runtime must not depend on Bench or optional llama.cpp integrations. Keep private environments and generated artifacts outside tracked product sources.

## Build and Testing Guidelines
- Before running tests for native/compiler changes, run `make` in the triton directory to rebuild triton. DO NOT RUN `make` if you only changed Python code or code in `python/triton_kernels`.
- For compiler changes, add tests in `python/test/` (pytest) or test (lit). Keep GPU-only tests in `python/test/unit/` or `python/test/gluon/`, name them `test_<feature>_<condition>`, and avoid creating new test files unless requested.
- Run pytest with `-s --tb=short`. Run a single test with `pytest file.py::test_name`.
- The build dir is given by `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
- Run lit from the build dir:  `cd BUILD_DIR; ninja triton-opt; lit -v test/<path>.mlir` (example: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir`).
- Lit tests can be run locally (no GPU required).
- Compiler crashes sometimes print an MLIR reproducer (external_resources / mlir_reproducer). Save the full MLIR + {-# ... #-} metadata to `/tmp/<file>.mlir`, then run `triton-opt /tmp/<file>.mlir --run-reproducer` to reproduce locally.
