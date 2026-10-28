# Pass、选择与软件流水

Pass 改写显式程序，消费已经建立的事实，把物理决定写入 IR，
并更新受改写影响的分析。

## 管线中的主要职责

| 阶段 | 工作 |
|---|---|
| 验证源合同 | 类型、轴、effects、Encoding 和 Level 初始化 |
| 导入目标表示 | 保留逻辑身份，建立存储访问 |
| 选择布局与供给 | 比较有限合法表示及共享消费者 |
| 物化 | 插入真实读取、packing、conversion 和局部目标操作 |
| 调度 | 按 SSA、内存与生命周期依赖放置操作 |
| 资源收口 | 分析活跃表示、局部存储与 spill/reload |
| 验证并生成代码 | 验证已选操作，再翻译 |

候选评估与正式编译复用同一套物化机制，
成本与资源分析因而覆盖真实物理工作，包括准备和其他消费者。

## 共享分析

`IntegerFacts` 提供范围与 signedness 证明；
`LogicalAxes`、`IndexRelations` 保留坐标身份；
`StorageBytes`、`StorageLifetime` 和 `PhysicalDependence`
描述 payload 访问、快照、alias 及 birth/handoff 约束。
这些分析位于目标指令选择之上。

## 软件流水深度

```python
configs = [
    triton.Config({"B": 64}, backend_options={"pipeline_depth": depth})
    for depth in (1, 2, 3, 4)
]
```

默认深度为 1。Scheduler 为当前程序选择合法 stage/order，
共享 expander 构造 prologue、steady state、epilogue、
跨阶段 SSA 和私有 buffer 版本。

展开后重新检查有效域、依赖和资源；buffer 版本数由生产与最后使用间的重叠决定。
源级 `tl.stage` 继续规定语义归属，物理流水深度规定执行调度。

## 查找实现

RISC-V 与 x86 的 `Backend` 目录组装各自管线，
目标 transforms 按算术、contraction、内存、控制和资源分类。
共享 full-tile 与软件流水变换位于 `third_party/weft/lib/Transforms`。

[RISC-V pass 规格](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/passes.md)
和 [x86 规格](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/x86.md)
提供 operation 级前置条件、改写过程与验证规则。
