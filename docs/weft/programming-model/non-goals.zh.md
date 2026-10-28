# 作者、编译器与应用的分工

判断一项优化应该落在哪一层，可以使用下面的职责表。

| 责任 | 所属位置 | 例子 |
|---|---|---|
| 算法与数值分解 | 作者 DSL | 将整数和分组，再应用尺度 |
| 逻辑分块与复用 | 作者 DSL | 让多个输出组共享同一 panel |
| 持久格式 | Encoding 与显式 builder | 模型初始化时交错权重 record |
| 局部表示 | 编译器 passes | 为矩阵指令打包同一逻辑 Value |
| 物理调度与资源 | 编译器 passes | 选择 LMUL、packet 宽度、流水深度与 spill |
| 实例划分与并发 | 宿主应用／grid | 将互不重叠的行分给 OpenMP 或 GGML worker |
| 存储所有权 | 宿主应用 | 分配 Tensor、workspace 和持久状态 |
| 比较协议 | Bench | 匹配 shape、参考、数值合同和计时范围 |

## 慢程序应该改哪里

如果源程序在内层 Level 中反复准备 panel，应在 DSL 中明确调整准备位置或算法组织。
如果 staged panel 的作用域已经合适，而生成代码仍重复 decode，应检查物理供给和资源 passes。
如果原生计算很快但应用调用昂贵，应检查 launch、复制与应用接入。

## 可观察合同

输入输出布局、持久存储、数值转换位置和初始化次数属于程序行为；
局部寄存器布局、临时 packing 与指令 issue 次序属于物理编译。
[详细边界规格](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/programming-model/non-goals.md)
记录 verifier 与 pass 实现使用的精确约束。
