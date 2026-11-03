# Install Triton–Weft 1.0

The product builds the `triton` distribution: language, native compiler,
CPU runtime, kernel library and C/C++ SDK. Weft 1.0.0 extends Triton 3.8.0;
its distribution version is `3.8.0+weft.1.0.0`.

## Install a released wheel

[Weft 1.0 downloads](https://github.com/Tianchen-ai/triton-weft/releases/tag/v1.0.0)
include Linux x86-64 wheels for Python 3.10 and 3.12, and a RISC-V wheel for Python 3.11.
Choose the matching architecture and Python ABI, then install the downloaded file:

```sh
python -m pip install ./triton-*.whl numpy
python -c 'import triton; print(triton.__weft_version__, triton.__version__)'
```

The wheels include the CPU compiler and native SDK. Source installation below supports
other Python versions and toolchain configurations.

## Install from source

Use native x86-64 or RISC-V Linux, Python 3.10+, GCC 12+, Git and the
dependencies in `python/requirements.txt`. Repository and extracted source
distribution use the same build; no `.git` directory is required.

```sh
git clone https://github.com/Tianchen-ai/triton-weft.git
cd triton-weft
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r python/requirements.txt numpy pytest
```

LLVM/MLIR must match revision
`5f07f818b51b786b0a87b4e514882600ecba112f`, recorded in
[`cmake/llvm-info.json`](https://github.com/Tianchen-ai/triton-weft/blob/main/cmake/llvm-info.json).
Build that SDK once:

```sh
LLVM_PROJECT_PATH="$PWD/.deps/llvm-project" \
LLVM_BUILD_PATH="$PWD/.deps/llvm-build" \
LLVM_INSTALL_PATH="$PWD/.deps/llvm" \
bash scripts/build-llvm-project.sh
```

Select the SDK to install Weft:

```sh
LLVM_SYSPATH="$PWD/.deps/llvm" bash scripts/build-weft.sh
python -c 'import triton; print("Weft", triton.__weft_version__, "on Triton", triton.__version__)'
python -m examples.weft.quickstart
```

An existing pinned SDK can be selected through `LLVM_SYSPATH`.
The helper finds Clang and OpenMP, selects host GCC and stores generated files
in `.build/`. Native RISC-V GNU builds use `-Wl,--no-relax`.
For editable installation run `bash scripts/build-weft.sh -e`.

## Explicit toolchain paths

GCC builds the extension; bundled Clang 23 compiles generated kernels.
Resource headers and OpenMP use the same dispatch ABI. Separately installed
components can be selected explicitly:

```sh
LLVM_SYSPATH=/opt/weft-llvm \
TRITON_WEFT_CLANG_ROOT=/opt/weft-clang \
TRITON_WEFT_OPENMP_LIBRARY=/opt/weft-openmp/lib/libomp.so \
TRITON_WEFT_OPENMP_INCLUDE_DIR=/opt/weft-openmp/include \
bash scripts/build-weft.sh
```

The Clang root contains `bin/clang`, `LICENSE.TXT` and
`lib/clang/23/include`; OpenMP headers include `omp.h`.
The build checks runtime entry points and native code generation.
The host also supplies the linker, libc development headers and startup files.

## Wheels and source distributions

With the same toolchain environment, build a wheel:

```sh
python -m pip wheel --no-build-isolation --no-deps . --wheel-dir dist
python -m pip install dist/triton-3.8.0+weft.1.0.0-*.whl
```

Choose the wheel matching architecture, Python ABI and system libraries.
The [host build workflow](https://github.com/Tianchen-ai/triton-weft/actions/workflows/build.yml)
provides a wheel artifact. The [release guide](../development/release.md)
describes source packages and the installed SDK.

## Native and cross compilation

Select cores with the [CPU launcher](hardware.md) before querying heterogeneous
CPU capabilities. JIT discovers the native target; AOT can supply one explicitly:

```python
options = {"toolchain": {
    "cc": "/opt/llvm/bin/clang",
    "target": "riscv64-unknown-linux-gnu",
    "sysroot": "/opt/riscv/sysroot",
    "sysroot_identity": "application-sdk-v1",
    "cflags": ["--gcc-install-dir=/opt/riscv/lib/gcc/riscv64-linux-gnu/14"],
}}
```

Use these options with `ASTSource` and `triton.compile`.
Python extension cross builds also specify
`TRITON_TARGET_PYTHON_INCLUDE_DIR`, `TRITON_TARGET_PYTHON_SOABI` and
`bdist_wheel --plat-name linux_riscv64`.
See [native deployment](../compiler/native-sdk.md).

## PyTorch

Install CPU PyTorch for the same Python ABI. Eager tensors pass directly to
kernels. For supported PyTorch 2.14 source interfaces, install the Inductor seam:

```sh
python -m triton.tools.pytorch
```

Then use `torch.compile(..., options={"cpu_backend": "triton"})`.
See [PyTorch and host composition](../integrations/overview.md).
