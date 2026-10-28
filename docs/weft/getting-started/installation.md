# Install Triton–Weft

Use a dedicated Python environment. This fork builds the `triton` distribution,
including the Python language, native compiler bindings, CPU runtime and application
SDK. Choose a wheel for your host architecture and Python ABI, or build current source.

## Published wheels

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://tianchen-ai.github.io/triton-weft/wheels/ 'triton==3.8.0+weft.2'
python -m pip install numpy
python -c "import triton; from triton import cpu; print(triton.__version__); print(cpu.target())"
```

[Release assets](https://github.com/Tianchen-ai/triton-weft/releases/tag/v3.8.0-weft.2)
contain Clang 23, its resource headers, OpenMP and the native SDK.

| Host | Python ABI | System libraries required by these wheels |
|---|---|---|
| Linux x86-64 | CPython 3.10 | glibc ≥ 2.34, GLIBCXX_3.4.30, CXXABI_1.3.13, libgcc, zlib |
| Linux RISC-V LP64D | CPython 3.11 / 3.12 / 3.14 | glibc ≥ 2.38, GLIBCXX_3.4.30, CXXABI_1.3.13, libgcc, libatomic |

JIT also uses the system linker, libc development headers and startup files.
Install the distribution's C/C++ development packages. On RISC-V, choose a NumPy
build matching the same interpreter and architecture. Use the
[core-selection launcher](hardware.md) before querying a heterogeneous CPU.

## Build current source

Clone the product and install its build dependencies:

```sh
git clone https://github.com/Tianchen-ai/triton-weft.git
cd triton-weft
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r python/requirements.txt build
```

The compiler, MLIR bindings and TableGen use the LLVM revision recorded in
[`cmake/llvm-info.json`](https://github.com/Tianchen-ai/triton-weft/blob/main/cmake/llvm-info.json).
Select that SDK and a host-native Clang 23/OpenMP bundle:

```sh
CC=/path/to/matched-llvm-sdk/bin/clang \
CXX=/path/to/matched-llvm-sdk/bin/clang++ \
LLVM_SYSPATH=/path/to/matched-llvm-sdk \
TRITON_WEFT_CLANG_ROOT=/path/to/native-clang-23 \
TRITON_WEFT_OPENMP_LIBRARY=/path/to/libomp.so \
TRITON_WEFT_OPENMP_INCLUDE_DIR=/path/to/openmp-headers \
TRITON_BUILD_DIR=/path/to/build \
TRITON_BUILD_PROTON=OFF \
python -m pip install --no-build-isolation .
```

`TRITON_WEFT_CLANG_ROOT` contains `bin/clang`, `LICENSE.TXT` and
`lib/clang/<version>/include`. OpenMP headers include `omp.h`; the runtime must
support Clang 23's dispatch ABI. The build checks these components.
`TRITON_BUILD_PROTON=OFF` selects a build without the optional profiler.

For an editable compiler checkout, use `pip install -e .` with the same environment.
For distribution, replace the final command with
`python -m pip wheel --no-build-isolation --no-deps . -w /path/to/wheels`.
The [CI host wheel](https://github.com/Tianchen-ai/triton-weft/actions/workflows/build.yml)
also provides a current build; its CPU toolchain is configured separately from the
bundled release wheels.

## Native and cross toolchains

Native JIT discovers its target after core selection and uses the packaged Clang.
An application can select an explicit toolchain through the launch options.
RISC-V uses the configured GCC installation's ABI runtime and startup files; Clang
performs kernel compilation. An explicit example is:

```python
options = {"toolchain": {
    "cc": "/opt/llvm/bin/clang",
    "target": "riscv64-unknown-linux-gnu",
    "sysroot": "/opt/riscv/sysroot",
    "sysroot_identity": "application-sdk-v1",
    "cflags": ["--gcc-install-dir=/opt/riscv/lib/gcc/riscv64-linux-gnu/14"],
}}
```

Give a target and `ASTSource` to `triton.compile` for an explicit cross build.
The [AOT chapter](../compiler/native-sdk.md) demonstrates a complete export and
application call. Python extension cross builds additionally specify
`TRITON_TARGET_PYTHON_INCLUDE_DIR`, `TRITON_TARGET_PYTHON_SOABI` and
`bdist_wheel --plat-name linux_riscv64` using the target interpreter's headers and ABI.

## PyTorch

Install a CPU PyTorch package for the same Python ABI. Eager tensors can be passed
directly to kernels. Graph integration uses the supported PyTorch 2.14 source
interfaces; apply the supplied seam once in that environment:

```sh
python -m triton.tools.pytorch
```

Then use `torch.compile(..., options={"cpu_backend": "triton"})`.
See [PyTorch and host composition](../integrations/overview.md).
