# 物理表示

一个逻辑 Value 可以大于单条机器向量或矩阵 fragment。
物理程序将其坐标映射到存储与执行，同时保留源程序的数值和生命周期关系。

## 物理组成

| 概念 | 作用 |
|---|---|
| Lane | 一次向量操作共同处理的元素 |
| Issue／time 维 | 通过连续操作执行的逻辑域部分 |
| Replica | 覆盖独立 free coordinate 的物理组 |
| Packet 或 fragment | 向量或矩阵指令消费的目标表示 |
| 局部存储 | 调用私有的快照、packing 或 spill |
| Conversion | 生产者与消费者表示间的显式转换 |
| 有效域 | 完整或尾部操作中有效的逻辑坐标 |

编译器可将 free coordinate 分成多个 packet，为多个消费者供应共享 Value，
也可在归约前提升 partial 位宽；这些表示选择保留作者的源级分组和作用域。

## 资源分析

目标 profile 提供向量宽度、架构寄存器事实、支持指令与策略预算。
当前 SSA 程序的分析决定活跃 Value、临时表示、packing 存储和 spill/reload 需求。

资源 pass 检查真实物化后的程序。当完整表示超出预算时，可以在有限合法候选中重新选择。
最终机器寄存器分配与指令调度由 LLVM 或目标 C 工具链完成。

## 两种目标表示

RISC-V 使用面向 RVV 与 IME 的自定义物理类型和操作；
x86 使用上游 MLIR vector、memref、SCF、arith 与 LLVM 操作，
配合自身的目标选择和资源分析。

继续阅读 [RISC-V IR](riscv-ir.md)、[x86 编译](x86.md)及
[完整物理机器合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/physical-machine.md)。
