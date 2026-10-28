# RISC-V Physical IR

RISC-V 物理方言描述同一 Canonical 程序如何使用 RVV 和可选矩阵扩展执行。
编译后通过 `compiled.asm["riscvir"]` 查看。

## 跟随一个编码 contraction

1. View／Encoding lowering 建立精确存储关系与有效坐标。
2. Layout passes 将 free/reduction coordinate 分配到 lane、issue 与 replica。
3. 供给 passes 物化读取、decode、局部 packing 与表示转换。
4. 指令选择形成合法 RVV 或 IME 局部操作。
5. 调度与资源 passes 完成 buffer 生命周期、活跃寄存器组和 spill。
6. 最终验证确认 emission 所需的决定均已明确。

尾部保留显式有效域；归约以对应 identity 填充无效物理坐标。
转换保持源程序的 signedness、位宽和轴身份。

## VLEN 与矩阵扩展

VLEN 是布局、资源分析和原生编译共同使用的目标事实。
同一逻辑程序在 VLEN 128、256、1024 上可以具有不同 packet 与 issue 数量。

IME1 和 IME2 增加局部矩阵合同与能力检查，其操作数、累加器和传输要求进入物理程序。
作者逐 contraction 指定的 `engine` 约束选择范围。

## 源码位置

- `Dialect/RISCV/IR`：自定义类型、操作与验证。
- `Target/RISCV/Conversion`：导入 Canonical 程序。
- `Target/RISCV/Transforms`：表示、算术、内存、contraction、控制与资源。
- `Target/RISCV/Extensions`：扩展指令合同。
- `Target/RISCV/Emission`：intrinsic C 与局部汇编。

[完整物理 operation 规格](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/riscv-ir.md)。
