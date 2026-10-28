# Triton–Weft 工具书

用 Triton 编写 CPU kernel，并进一步表达影响实现的数据存储、数值分组和复用关系。
Weft 将这些关系加入语言，经过共同编译主链，在 RISC-V 与 x86 上形成原生程序。

## 按任务开始

| 接下来要做的事 | 阅读入口 |
|---|---|
| 跑通第一个 kernel | [安装](getting-started/installation.md)，然后[快速上手](getting-started/quickstart.md) |
| 理解扩展的编程模型 | [模型](programming-model/index.md)、[Encoding](dsl/encoding-and-artifacts.md)、[Level 与状态](dsl/values-and-levels.md) |
| 选择核心、ISA 与矩阵计算 | [目标与 engine](getting-started/hardware.md) |
| 调优和部署 | [JIT 与 autotune](compiler/runtime.md)、[C++](compiler/native-sdk.md)、[llama.cpp](integrations/llama-cpp.md) |
| 复现论文实验 | [Bench 工作流](experiments/reproduce.md)、[图表与测量设置](experiments/results.md) |
| 开发编译器 | [编译管线](compiler/index.md)、[passes](compiler/passes.md)、[扩展指南](development/extensions.md) |

页面右上方的语言选择器可切换到对应的英文或中文章节。
两种版本的代码统一使用 `import triton.language as tl`。

[![共同前端和目标编译](assets/paper/compiler.png)](assets/paper/compiler.png)

## 从源程序到执行

1. 使用 `@triton.jit(model="weft")` 编写普通 Triton kernel。
2. 在算法需要显式关系的地方组合 View、Encoding 或 Level。
3. 通过 `kernel[grid](...)` 调用，Triton 完成特化、编译、缓存与执行。
4. 查看 `CompiledKernel.asm`，调优有限配置，或导出原生产物。

[示例教程](getting-started/examples.md)按概念逐步展开，每个示例都包含输入准备、
可调用 kernel 和宿主应用。

## 论文实验结果

使用 Triton–Weft 与 [Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench)
复现论文中的算子和模型实验。数据集覆盖 553 个算子—平台案例和九组模型配置。

[![代表算子与全部匹配算子比较](assets/paper/operators.png)](assets/paper/operators.png)

| 平台 | 匹配原生比较 | 几何平均加速 |
|---|---:|---:|
| SG2044 | 132 | **1.60×** |
| K1 | 136 | **1.55×** |
| K3 A100 | 135 | **1.82×** |
| Xeon Platinum 8470Q | 140 | **1.56×** |

倍率为对应参考耗时除以 Triton–Weft 耗时。

[![九组模型配置，prefill 与 decode 分别展示](assets/paper/models.png)](assets/paper/models.png)

18 个模型阶段覆盖 Llama 2 7B、DeepSeek 8B 和 Qwen 2.5 14B，均使用八线程。
K3 的 CPU–RVV 和厂商 IME2 采用各自的参考策略。
查看[复现步骤](experiments/reproduce.md)，或直接阅读
[图表、完整 CSV 和测量设置](experiments/results.md)。

## 文档与源码

工具书说明使用方法和编译器职责；完整数值、IR 与生命周期合同保存在
[设计规格](https://github.com/Tianchen-ai/triton-weft/tree/main/docs/specification)。
[源码地图](development/upstream.md)连接规格与实现。
[Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench)
集中管理比较用例、参考库与实验结果。
