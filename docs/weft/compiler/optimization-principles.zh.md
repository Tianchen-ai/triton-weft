# 优化真实物理工作

先看生成程序中的重复读取、decode、地址计算、conversion、归约、spill 和等待，
再找到使更好实现合法的源程序关系。

## 建议的定位顺序

1. 阅读作者程序的 blocking、state 和 stage 位置。
2. 在 Canonical IR 中核对数值与复用关系。
3. 查看物理布局、真实访问和 conversion。
4. 阅读资源结论与原生代码。
5. 使用 Bench 和匹配的原生参考测量受影响用例。

## 有界选择

目标提供有限合法候选，例如向量分组、packet 宽度、fragment 表示和供给策略。
显式作者或 backend 配置固定对应维度，每个候选在选择域内物化并评估一次。

成本覆盖准备、packing、地址、load/store、decode、conversion、算术、归约和
显式 spill/reload。静态次数使用已知 extent，动态工作保留相应符号量；
最终数值、effect 与资源合法性决定候选能否采用。

## 作者改进与编译器改进

更合适的 source block 或持久布局属于作者程序改进；
同一程序中更好的寄存器布局和共享供给属于编译器改进。
DSL 变化时，依次比较旧编译器／旧 DSL、新编译器／旧 DSL、
新编译器／新 DSL，使两者贡献清晰可见。

[Bench 复现](../experiments/reproduce.md)说明输入和计时匹配。
[完整优化合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/optimization-principles.md)
定义候选边界与资源反馈。
