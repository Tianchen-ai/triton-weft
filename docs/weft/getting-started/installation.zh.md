# 安装 Triton–Weft 1.0

产品构建 `triton` 发行包，包含语言、原生编译器、CPU 运行时、内核库与 C/C++ SDK。
Weft 1.0.0 扩展自 Triton 3.8.0，发行包版本为 `3.8.0+weft.1.0.0`。

## 安装正式发布的 wheel

[Weft 1.0 下载页](https://github.com/Tianchen-ai/triton-weft/releases/tag/v1.0.0)
提供 Python 3.10／3.12 的 Linux x86-64 wheel，以及 Python 3.11 的 RISC-V wheel。
选择匹配 CPU 架构和 Python 版本的文件，然后安装：

```sh
python -m pip install ./triton-*.whl numpy
python -c 'import triton; print(triton.__weft_version__, triton.__version__)'
```

wheel 包含 CPU 编译器和原生 SDK。其他 Python 版本或工具链配置可按下文从源码构建。

## 从源码安装

使用原生 x86-64 或 RISC-V Linux、Python 3.10+、GCC 12+、Git，
以及 `python/requirements.txt` 中的依赖。
仓库与解压后的源码发行包使用相同构建方式，无需 `.git` 目录。

```sh
git clone https://github.com/Tianchen-ai/triton-weft.git
cd triton-weft
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r python/requirements.txt numpy pytest
```

LLVM/MLIR 必须对应
`5f07f818b51b786b0a87b4e514882600ecba112f`，
记录见 [`cmake/llvm-info.json`](https://github.com/Tianchen-ai/triton-weft/blob/main/cmake/llvm-info.json)。
SDK 只需构建一次：

```sh
LLVM_PROJECT_PATH="$PWD/.deps/llvm-project" \
LLVM_BUILD_PATH="$PWD/.deps/llvm-build" \
LLVM_INSTALL_PATH="$PWD/.deps/llvm" \
bash scripts/build-llvm-project.sh
```

选择 SDK 安装 Weft：

```sh
LLVM_SYSPATH="$PWD/.deps/llvm" bash scripts/build-weft.sh
python -c 'import triton; print("Weft", triton.__weft_version__, "on Triton", triton.__version__)'
python -m examples.weft.quickstart
```

已有固定版本 SDK 时直接指定 `LLVM_SYSPATH`。
脚本查找 Clang 和 OpenMP，使用宿主 GCC，将生成文件放入 `.build/`。
原生 RISC-V GNU 构建使用 `-Wl,--no-relax`。
开发时运行 `bash scripts/build-weft.sh -e` 进行 editable 安装。

## 显式选择工具链

GCC 编译扩展，随包的 Clang 23 编译生成的 kernel。
资源头和 OpenMP 使用相同 dispatch ABI。组件分开安装时可指定：

```sh
LLVM_SYSPATH=/opt/weft-llvm \
TRITON_WEFT_CLANG_ROOT=/opt/weft-clang \
TRITON_WEFT_OPENMP_LIBRARY=/opt/weft-openmp/lib/libomp.so \
TRITON_WEFT_OPENMP_INCLUDE_DIR=/opt/weft-openmp/include \
bash scripts/build-weft.sh
```

Clang 根目录包含 `bin/clang`、`LICENSE.TXT` 与 `lib/clang/23/include`，
OpenMP 头目录包含 `omp.h`。构建检查运行库入口与原生代码生成器；
宿主系统另需链接器、libc 开发头和启动文件。

## wheel 与源码发行包

保留相同工具链环境变量，构建 wheel：

```sh
python -m pip wheel --no-build-isolation --no-deps . --wheel-dir dist
python -m pip install dist/triton-3.8.0+weft.1.0.0-*.whl
```

wheel 需匹配架构、Python ABI 和系统库。
[宿主构建工作流](https://github.com/Tianchen-ai/triton-weft/actions/workflows/build.yml)
提供 wheel artifact；[发布指南](../development/release.md)介绍源码包和安装后的 SDK。

## 原生与交叉编译

异构 CPU 先通过 [CPU launcher](hardware.md) 选择核心，再查询能力。
JIT 识别原生目标，AOT 可以显式提供目标与工具链：

```python
options = {"toolchain": {
    "cc": "/opt/llvm/bin/clang",
    "target": "riscv64-unknown-linux-gnu",
    "sysroot": "/opt/riscv/sysroot",
    "sysroot_identity": "application-sdk-v1",
    "cflags": ["--gcc-install-dir=/opt/riscv/lib/gcc/riscv64-linux-gnu/14"],
}}
```

将这些选项用于 `ASTSource` 与 `triton.compile`。
Python 扩展交叉构建还指定 `TRITON_TARGET_PYTHON_INCLUDE_DIR`、
`TRITON_TARGET_PYTHON_SOABI` 和 `bdist_wheel --plat-name linux_riscv64`。
继续阅读[原生部署](../compiler/native-sdk.md)。

## PyTorch

安装与 Python ABI 匹配的 CPU PyTorch，Eager Tensor 可直接传给 kernel。
对于支持的 PyTorch 2.14 源码接口，安装 Inductor 接缝：

```sh
python -m triton.tools.pytorch
```

随后使用 `torch.compile(..., options={"cpu_backend": "triton"})`。
继续阅读 [PyTorch 与宿主组合](../integrations/overview.md)。
