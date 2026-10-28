# 安装 Triton–Weft

在独立 Python 环境中安装。本 fork 构建名为 `triton` 的发行包，包含 Python 语言、
原生编译绑定、CPU runtime 和应用 SDK。按宿主架构与 Python ABI 选择 wheel，
也可以从当前源码构建。

## 已发布的 wheel

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://tianchen-ai.github.io/triton-weft/wheels/ 'triton==3.8.0+weft.2'
python -m pip install numpy
python -c "import triton; from triton import cpu; print(triton.__version__); print(cpu.target())"
```

[Release 资产](https://github.com/Tianchen-ai/triton-weft/releases/tag/v3.8.0-weft.2)
包含 Clang 23、resource headers、OpenMP 和原生 SDK。

| 宿主 | Python ABI | 这些 wheel 所需的系统库 |
|---|---|---|
| Linux x86-64 | CPython 3.10 | glibc ≥ 2.34、GLIBCXX_3.4.30、CXXABI_1.3.13、libgcc、zlib |
| Linux RISC-V LP64D | CPython 3.11／3.12／3.14 | glibc ≥ 2.38、GLIBCXX_3.4.30、CXXABI_1.3.13、libgcc、libatomic |

JIT 还使用系统 linker、libc 开发头文件和启动文件，请安装发行版的 C/C++ 开发包。
RISC-V 的 NumPy 需要匹配解释器与架构。异构 CPU 应先使用[选核启动器](hardware.md)，
再查询当前目标。

## 从当前源码构建

克隆产品仓库并安装构建依赖：

```sh
git clone https://github.com/Tianchen-ai/triton-weft.git
cd triton-weft
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r python/requirements.txt build
```

编译器、MLIR bindings 和 TableGen 共用
[`cmake/llvm-info.json`](https://github.com/Tianchen-ai/triton-weft/blob/main/cmake/llvm-info.json)
记录的 LLVM revision。选择匹配的 SDK，以及宿主原生 Clang 23／OpenMP：

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

`TRITON_WEFT_CLANG_ROOT` 包含 `bin/clang`、`LICENSE.TXT` 和
`lib/clang/<version>/include`。OpenMP 头文件包含 `omp.h`，
runtime 应支持 Clang 23 的 dispatch ABI；构建时会检查这些组件。
`TRITON_BUILD_PROTON=OFF` 选择不包含可选 profiler 的构建。

开发时可在相同环境使用 `pip install -e .`；生成发行包则把最后一行换为
`python -m pip wheel --no-build-isolation --no-deps . -w /path/to/wheels`。
[CI host wheel](https://github.com/Tianchen-ai/triton-weft/actions/workflows/build.yml)
也提供当前构建，其 CPU 工具链需单独配置；Release wheel 随包提供工具链。

## 本机与交叉工具链

本机 JIT 在选核后发现目标，并使用随包 Clang。应用可以通过启动参数选择显式工具链。
RISC-V 使用指定 GCC 安装中的 ABI runtime 与启动文件，kernel 由 Clang 编译。例如：

```python
options = {"toolchain": {
    "cc": "/opt/llvm/bin/clang",
    "target": "riscv64-unknown-linux-gnu",
    "sysroot": "/opt/riscv/sysroot",
    "sysroot_identity": "application-sdk-v1",
    "cflags": ["--gcc-install-dir=/opt/riscv/lib/gcc/riscv64-linux-gnu/14"],
}}
```

显式交叉构建使用 `triton.compile`，传入目标和 `ASTSource`。
[AOT 章节](../compiler/native-sdk.md)给出完整导出与应用调用。
交叉构建 Python extension 时，还需按目标解释器的 headers 与 ABI 指定
`TRITON_TARGET_PYTHON_INCLUDE_DIR`、`TRITON_TARGET_PYTHON_SOABI`，
以及 `bdist_wheel --plat-name linux_riscv64`。

## PyTorch

安装与当前 Python ABI 匹配的 CPU PyTorch 包。Eager Tensor 可以直接传给 kernel。
图接入使用受支持的 PyTorch 2.14 源码接口，在该环境执行一次：

```sh
python -m triton.tools.pytorch
```

随后使用 `torch.compile(..., options={"cpu_backend": "triton"})`。
完整示例见 [PyTorch 与宿主组合](../integrations/overview.md)。
