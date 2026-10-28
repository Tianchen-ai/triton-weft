# 局部指令合同

局部 target operation 描述一段闭合机器工作：
typed operand、坐标覆盖、数值行为、传输、effects 与资源。
物理 passes 检查这些合同后选择对应操作。

## 合同内容

| 内容 | 需要回答的问题 |
|---|---|
| 类型与位宽 | 指令能否表示源程序的乘积和 accumulator？ |
| 轴与有效域 | 覆盖哪些 free/reduction coordinate？ |
| 操作数布局 | 消费哪个 packet、寄存器组或矩阵 fragment？ |
| 数值语义 | 是否保持 signedness、回绕、舍入和归约边界？ |
| Effects 与传输 | 读写哪些字节或局部存储版本？ |
| 资源与能力 | 需要哪些寄存器、scratch 字节和 ISA 扩展？ |

对于 grouped integer operation，共同整数事实证明 widening 或 saturating 机器指令是否合法；
对于矩阵操作，物理操作数布局和 accumulator 表示应与所选合同一致。

## 扩展模块

IME 或 AMX 操作沿普通物理编译过程接入：
能力声明 → 合法性 → 选择 → 物化 → 验证 → emission。
外围遍历和源级 state 仍由周围程序表示。

[硬件扩展指南](../development/extensions.md) ·
[完整 RISC-V 局部操作规格](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/leaves.md)。
