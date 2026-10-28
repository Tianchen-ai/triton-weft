# Value、计算层与状态

Value 具有元素类型、逻辑 shape 和轴身份；Level 为一组计算规定 domain、partition
与生命周期。两者共同让复用和累加成为作者程序的一部分。

## 四个基本构造

| 写法 | 含义 |
|---|---|
| `tl.load(region)` | 在有序程序当前位置读取一个 Value |
| `tl.stage(expression)` | 建立本层持有、供后代复用的只读值 |
| `tl.state(dtype, shape, init=...)` | 初始化归本层所有的可更新状态 |
| `tl.store(region, value)` | 写入指定逻辑 region |

改变 state 的声明层，会改变初始化时机；改变 stage 的声明层，会改变共享它的后代范围。
这些都由作者在源码中明确选择。

## 选择计算域

```python
with tl.level.rows(M, group=1) as row:
    values = tl.stage(tl.load(X[row]))
    total = tl.state(tl.f32, [1], init=0)
    with tl.level.blocks(N, extent=32) as block:
        part = values[:, block]
        total += tl.reduce(part * part, axis="n", op="add")
    tl.store(Y[row], total)
```

对于 65 元素的一行，内层 Level 依次覆盖 32、32、1 个有效元素，
累加结果在三个实例间都归外层 row 持有。
`rows/cols(..., group=...)` 表达逻辑 cohort，
`tiles/blocks(..., extent=...)` 划分轴，`subtiles(parent, extent=...)` 细分父域。

目标可用多次物理 issue 执行同一个 cohort；逻辑归属和初始化次数保持作者声明。

## 初始化与可见性

进入 Level 后依次执行 state 初始化、stage 初始化和 body。
每个初始化表达式可以使用进入该层前的值、当前 Level point，以及本初始化区域先前定义的值。

把 staged 输入直接写在初始化中：

```python
with tl.level.rows(M, group=1) as row:
    values = tl.stage(tl.load(X[row]))
    # 后代消费 values。
```

这样就明确了读取的作用域。body 局部临时值归 body 所有；初始化捕获进入该层时的 SSA 环境。

## 更新与 handoff

普通赋值表达状态变化，例如 `total += contribution`、`m = new_m`，
或 `o = o * alpha + contribution`。Canonical IR 通过 Level result
传递全部更新值；多个 carry、普通分支和嵌套循环使用相同 SSA 关系。

普通 `for`、`while` 和 `if` 表达有序标量控制，适用于索引遍历或依赖搜索。
算法需要明确逻辑轴时，使用 shaped Value 表达。

## 源级 stage 与软件流水

`tl.stage` 规定值的归属和复用；
`backend_options={"pipeline_depth": 3}` 请求该程序的物理软件调度。
scheduler 检查依赖，expander 为跨阶段的值与私有 buffer 建立版本。
[流水章节](../compiler/passes.md)解释具体过程。

[完整 Value 与 Level 语义](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/dsl/values-and-levels.md)。
