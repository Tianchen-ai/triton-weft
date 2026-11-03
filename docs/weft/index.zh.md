---
hide:
  - toc
---

# Triton–Weft 1.0

**组合专家级 CPU 内核，由编译器构造机器执行。**

算子接口留下了许多决定性能的选择：展开压缩权重，还是共享一次准备的区域？
局部乘积如何分组？状态跨越哪些更新？这些选择各有自己的作用范围。
CPU 内核开发需要直接改变、组合它们，并让编译器重新构造对应的机器执行。

[![论文图 1：从算子的实现选择，到 Weft 程序及原生 CPU 执行](assets/paper/overview.svg)](assets/paper/overview.svg)

*Weft 补上这一抽象：源程序表达实现关系，编译器构造机器映射。*

Weft 将存储字段、带类型的局部算术、共享准备和初始化状态表达为可组合的源程序对象。
它通过扩展 Triton 实现，经过 Canonical IR 保留这些选择，为 **x86 与 RISC-V**
构造向量或矩阵执行。同一模型支持量化与稠密投影、归一化、在线注意力和流式统计。

[安装](getting-started/installation.md) · [编写第一个内核](getting-started/quickstart.md)
· [Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench)
· [完整论文结果](https://tianchen-ai.github.io/triton-weft-bench/zh-CN/paper/gallery/)

## 从完整程序开始

[安装 Weft](getting-started/installation.md)，运行[快速上手](getting-started/quickstart.md)：
程序通过带类型的 View、共享输入和跨块更新的状态计算每行平方和。
随后运行[完整示例](getting-started/examples.md)中的 Q4_K 投影、在线注意力、
Welford 统计和 GEMM。

| 接下来要做的事 | 阅读入口 |
|---|---|
| 表达存储与数值选择 | [编程模型](programming-model/index.md)、[Encoding](dsl/encoding-and-artifacts.md)、[运算](dsl/functions-and-operations.md) |
| 组合共享与状态 | [Level 与状态](dsl/values-and-levels.md)、[模型程序](dsl/programs.md) |
| 选择核心与计算引擎 | [目标与 engine](getting-started/hardware.md) |
| 调优与部署 | [JIT 与自动调优](compiler/runtime.md)、[原生 C/C++](compiler/native-sdk.md)、[llama.cpp](integrations/llama-cpp.md) |
| 复现实验 | [Bench 工作流](experiments/reproduce.md)、[结果与数据](experiments/results.md) |
| 开发编译器 | [编译管线](compiler/index.md)、[passes](compiler/passes.md)、[源码地图](development/upstream.md) |

页面语言选择器切换到对应的英文或中文章节，两种版本统一使用
`import triton.language as tl`。

## 从源程序关系到原生执行

[![Triton 前端、Canonical IR 与 x86/RISC-V 原生编译](assets/paper/compiler.svg)](assets/paper/compiler.svg)

作者使用 `@triton.jit(model="weft")`，通过 `kernel[grid](...)` 调用。
字段、数值分组、共享和状态在编译过程中保持显式；
目标 pass 决定向量布局、矩阵片段与局部存储。
运行时支持 JIT 缓存、自动调优、原生导出和应用调用。

## 实测结果

在 553 个算子—平台案例中的 **543 项匹配对照**上，生成内核相对优化的 GGML、oneDNN
参考实现取得 **1.32–1.83×** 的分平台几何平均加速。完整分布覆盖编码计算、
格式转换、稠密计算和行算子。

[![全部匹配案例的加速累积分布](assets/paper/operators-ecdf.svg)](assets/paper/operators-ecdf.svg)

| 平台 | 匹配比较 | 几何平均加速 |
|---|---:|---:|
| SG2044 | 132 | **1.60×** |
| K1 | 136 | **1.54×** |
| K3 A100 | 135 | **1.83×** |
| Xeon Platinum 8470Q | 140 | **1.32×** |

完整推理覆盖八组模型—平台部署、18 项阶段比较。
SG2044 使用 64 个应用线程；K1、K3 和 x86 使用八线程。
K3 同时展示 CPU 向量参考与厂商 IME2 部署比较。

[![完整 prefill 与 decode 测量](assets/paper/models.svg)](assets/paper/models.svg)

这些性能来自源程序的控制及编译器对它们的利用。[精选结果](experiments/results.md)
展示五路线控制任务、格式复用，以及固定源语义后的物理编译机制。
**[Bench 完整图集](https://tianchen-ai.github.io/triton-weft-bench/zh-CN/paper/gallery/)**
收录论文全部结果图，从硬件微内核、线程扩展到源码增量与编译器研究，并链接十六份 CSV。

[使用 Bench 复现](experiments/reproduce.md)：选择工作负载、安装后的内核、参考与 CPU，
Bench 统一准备输入、检查输出，测量匹配的原生调用或完整 llama.cpp 阶段。

## 版本与实现

Weft **1.0.0** 扩展自 Triton **3.8.0**，发行包版本为
`3.8.0+weft.1.0.0`，Python API 分别提供两个版本标识。
[发布指南](development/release.md)介绍源码包与验证；
[设计规格](https://github.com/Tianchen-ai/triton-weft/tree/main/docs/specification)
给出完整语言与编译器合同。
