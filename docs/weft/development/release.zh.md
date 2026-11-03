# Weft 1.0 发布

Weft **1.0.0** 是编程模型、编译器与 CPU 运行时版本，
基于上游 Triton **3.8.0**。

| 接口 | 版本 |
|---|---|
| Python 发行包 | `triton==3.8.0+weft.1.0.0` |
| Weft API | `triton.__weft_version__ == "1.0.0"` |
| 上游 API 基础 | `triton.__version__ == "3.8.0"` |
| 原生 CMake package | `find_package(Weft 1.0 CONFIG REQUIRED)` |
| 比较项目 | `triton-weft-bench==1.0.0` |

## 源码与发行包

公开源码包含完整 Triton fork、共同层与目标编译、运行时/SDK、
可安装的 `triton.kernels.weft`、示例、测试、双语工具书、许可证和十四份结果表。
独立的 [Bench 项目](https://github.com/Tianchen-ai/triton-weft-bench)
负责参考适配、工作负载目录与测量。

使用 `python -m build --sdist --no-isolation` 构建源码发行包，
解压后按照[安装章节](../getting-started/installation.md)操作，
无需 Git 历史或另一份工作树。
固定 LLVM/MLIR SDK 是外部构建依赖，
可用 `scripts/build-llvm-project.sh` 生成；
wheel 包含 Clang 资源和原生运行时/SDK。

## 验证安装

```sh
python -c 'import triton; print(triton.__weft_version__, triton.__version__)'
python -m examples.weft.quickstart --dump-ir /tmp/weft-ir --export /tmp/weft-native
python -m examples.weft.kernels.projection
python -m examples.weft.kernels.attention
python -m examples.weft.kernels.welford
python -m examples.weft.kernels.gemm
python -m pytest python/test/weft -s --tb=short
```

示例构造输入并检查完整输出。
[原生 SDK](../compiler/native-sdk.md)提供导出 ABI 的 C/C++ 调用，
[目标与 engine](../getting-started/hardware.md)解释核心选择与矩阵指令。

## 工具书与实验材料

```sh
python -m pip install -r third_party/weft/requirements-docs.txt
mkdocs build --strict -f mkdocs.yml
```

此命令同时构建中英文工具书。
[结果章节](../experiments/results.md)关联认可的图件、未舍入 CSV 与来源记录。
新测量使用已安装产品和 Bench；本次发布保持论文已冻结的工作负载与测量值。
