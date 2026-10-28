# Canonical Kernel IR

Canonical IR 是作者数值程序的 typed 表示，由共同 Triton 前端与 Weft semantic builder
在进程内构造。通过 `compiled.asm["weftir"]` 可以查看。

## 保存的信息

| 源程序概念 | Canonical 信息 |
|---|---|
| View 与 Encoding | 逻辑 shape、字段映射、access 与 alias 合同 |
| Value | 元素类型、signedness、shape 与轴身份 |
| Level | domain、partition、初始化区域、body 与 carried result |
| state/stage | 诞生作用域、初始化和合法消费者 |
| 算术 | operation 边界、grouped 位宽、转换和归约轴 |
| 内存 | 存储来源、offset、mask、填充值和有序 effects |
| Contraction | typed operand、free/reduction axis 与作者的 `engine` 约束 |

Encoding 字段保留到存储的坐标关系，grouped partial 保留累加类型，
staged Value 保留生命周期与消费者。这些事实供两个目标后端及框架接口共同使用。

## 验证与分析

verifier 检查类型和轴一致性、初始化可见性、carry/handoff 类型、
存储映射，以及各操作的数值合同。共享分析推导整数事实、索引关系、
存储边界、alias 和依赖；改写 pass 会重建被修改影响的事实。

普通 Triton tensor 语义经过相同前端。标准 dot 保留位置轴矩阵约定，
显式具名轴运算保留逻辑身份。

## 实现位置

Canonical 定义位于 `third_party/weft/include/Weft/Dialect/Kernel`。
`lib/Dialect/Kernel/IR` 按类型、Encoding、控制、内存与数值验证分工；
`lib/Analysis` 保存 RISC-V 和 x86 共用的分析。

[完整 operation 与 verifier 规格](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/dsl/canonical-ir.md)。
