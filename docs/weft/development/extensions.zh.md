# 增加硬件扩展

Weft 分别组织能力事实、局部指令合同、物理变换、emission 和 runtime 检查。
新增 RISC-V 扩展或 x86 指令族时沿用这些接口。

## 机器事实与局部合同

描述 ISA feature、操作数／accumulator 类型、支持维度、对齐与资源，
命名 CPU profile 展开为这些事实。
每个局部操作规定数值行为、逻辑覆盖、操作数表示、effects、传输和临时资源，
合法性检查复用共同整数、轴与存储证明。

遍历、source 分组、state 和复用作用域由作者程序提供，
扩展实现物理程序中的局部部分。

## 接入管线

| 步骤 | 职责 |
|---|---|
| 能力 | 平台描述与本机发现 |
| 合法性 | typed 数值、布局和 effect 要求 |
| 选择 | 满足 contraction engine 与有限合法选项 |
| 物化 | 传输、packing、conversion 和已选操作 |
| 资源 | 活跃寄存器／fragment、局部 buffer 与调度 |
| Emission | 所选 intrinsic、汇编或 LLVM 操作 |
| Runtime | 在实际 worker 检查 artifact 要求 |

RISC-V 模块位于 `Target/RISCV/Extensions/SpacemitIME1` 和 `SpacemitIME2`；
x86 AMX 归 contraction、platform 模块。
共同数值与存储语义保留在 Canonical IR 和 `lib/Analysis`。

## 执行作者程序

编写使用该操作的 DSL，经过普通 Triton 前端编译。
Bench 检查数值边界、尾部与真实目标执行，
随后将受影响工作负载与匹配参考比较，并查看实际 artifact 扩展及调用。

[完整扩展合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/development/extensions.md) ·
[源码地图](upstream.md) · [Bench 工作流](../experiments/reproduce.md)。
