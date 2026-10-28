# 上游关系与源码地图

本仓库 fork 自 [triton-lang/triton](https://github.com/triton-lang/triton)，
基于 `v3.8.0`，以 `main` 为产品分支，并保留上游历史。

```sh
git remote add upstream https://github.com/triton-lang/triton.git
git diff v3.8.0 -- python/triton/runtime python/triton/compiler python/triton/language
git diff v3.8.0 -- CMakeLists.txt setup.py python/src
```

| 模块 | 职责 |
|---|---|
| `language/core.py`、`language/semantic.py` | 标准 builtin 与逐 contraction 的 CPU 约束 |
| `compiler/code_generator.py`、`compiler/weft/` | 共同 AST、作用域、SSA；Encoding／Level 分析与 Canonical builder |
| `runtime/jit.py`、`runtime/autotuner.py` | 模型目标、特化、缓存与作者配置调优 |
| `backends/weft/` | 目标参数、物理管线、工具链阶段与 CPU driver |
| `python/src/weft.cc` | 进程内 bindings 与能力查询 |
| `third_party/weft/lib/Dialect/Kernel/IR` | 类型、Encoding、控制、内存和数值合同 |
| `Dialect/Kernel/Transforms` | 两目标 lowering 前的 source 特化 |
| `lib/Analysis` | 共同整数、轴、索引、存储、生命周期与依赖分析 |
| `lib/Transforms` | 共享 SCF full-tile 和有界软件流水展开 |
| `lib/Target/RISCV`、`Target/X86` | 各自 Backend、Platform、Analysis、Conversion、Transforms、Emission |
| `lib/Runtime` | 原生 ABI、库生命周期和 worker 检查 |
| `python/triton/integrations`、`integrations/llama.cpp` | 应用合同与 GGML 调用点 |
| `cpu.py`、`tools/cpu.py` | 目标构造、同质核心发现与进程启动 |
| `CMakeLists.txt`、`setup.py` | 统一 SDK、Triton 发行包与原生工具链 |

标准 GPU 路线与上游 tests 保留在 fork。
Weft 优化 kernel、数值检查、参考配方与性能输出由
[Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench)管理。

## 目标目录

`Backend` 组装管线，`Platform` 描述 ISA 与执行能力，
`Analysis` 分析目标表示，`Conversion` 导入 Canonical，
`Transforms` 按内存、算术、contraction、控制和资源物化程序，
`Emission` 翻译物理决定。公共头文件对应
`include/Weft/Target/RISCV` 与 `X86`。

`Dialect/RISCV/IR` 定义 RVV／IME 类型和操作；
x86 复用上游 vector、memref、SCF、arith 和 LLVM 方言，
由 `Target/X86/IR/Dialects.cpp` 注册。
Canonical type/axis 查询位于 `Dialect/Kernel/IR/TypeQueries`，
索引证明位于 `Analysis/IndexRelations`；寄存器组与 packet 压力归目标分析。

## 构建与维护

`Build Triton and Weft` 检查 host wheel 与双语工具书并发布 Pages。
上游 wheel/PyPI、GPU 文档任务使用其专用 runner 和发布凭据。
安装本 fork 的 `triton` wheel 获得前端和编程模型扩展。
工具书位于 `docs/weft`，完整合同位于 `docs/specification`。

上游 Triton 保留 MIT，Weft 保留 Apache-2.0；
根目录 NOTICE 和 `third_party/weft/licenses` 保存归属。
