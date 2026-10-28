# 表达决定实际工作的程序关系

一个 CPU kernel 往往同时具有几种重要边界：存储 record、整数乘积分组、输出 tile、
可复用 panel，以及跨 panel 更新的 accumulator。Weft 让作者分别表达这些边界及其关系，
编译器再利用这些信息，在具体向量或矩阵硬件上实现程序。

## 四种相互关联的选择

| 选择 | 源程序构造 | 例子 |
|---|---|---|
| 字节怎样表示逻辑数据 | Encoding 与 View | Q4 record 保存打包数值和独立的 scale 字段 |
| 哪些运算组成一次数值贡献 | typed arithmetic、grouped MAC、widen/reduce | 四个乘积先在 I16 中累加，再提升为 I32 |
| 哪些消费者共享一个值 | Level 与 stage | 多个输出组消费同一份 staged 权重 panel |
| 一条更新链存活多久 | Level 与 state | F32 accumulator 跨 panel 内各 block 保留 |

[![Q4_K 投影中独立的存储、分组和状态边界](../assets/paper/model.svg)](../assets/paper/model.svg)

论文的 Q4_K 投影中，**A** 是共享权重 panel，**B** 是每个输出组的 F32 state，
**C** 是单个 256 元素 block 的 I32 state，**D** 是 grouped I16 partial。
改变共享 panel 的作用域会改变准备工作的重复次数；改变 partial 位宽会改变算术和转换。
这些选择都保留在源程序中。

## 一个完整的小例子

[Level 教程](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/language/levels.py)
计算每行的平方和：

```python
from __future__ import annotations
import triton
import triton.language as tl

@triton.jit(model="weft")
def row_sum_squares(X: tl.View[tl.f32, (M, N)],
                    Y: tl.View[tl.f32, (M,)], B: tl.constexpr = 32):
    with tl.level.rows(M, group=1) as row:
        values = tl.stage(tl.load(X[row]))
        total = tl.state(tl.f32, [1], init=0)
        with tl.level.blocks(N, extent=B) as block:
            part = values[:, block]
            total += tl.reduce(part * part, axis="n", op="add")
        tl.store(Y[row], total)
```

行这一层持有 `values` 和 `total`，内层 block 持续更新同一 state，最后的部分 block
也属于这条更新链。编译器可将逻辑行分配到多个机器寄存器和 issue 中，同时保持初始化与更新边界。

## 编译器怎样使用模型

| 源程序信息 | 支持的物理决定 |
|---|---|
| 字段位映射和 signedness | decode、gather、unpack 与合法机器位宽 |
| free axis 和 reduction axis | lane 放置、packet 形状和归约结构 |
| 共享 Value 及其消费者 | 寄存器驻留、局部 packing 或合法重读 |
| state 生命周期与 handoff | accumulator 表示、carry 存储与调度 |
| `engine` 和目标能力 | 为每个 contraction 选择 RVV／IME 或 SIMD／AMX 操作 |

作者控制逻辑分块、数值支路、持久布局和作用域；编译器在这些选择内决定物理实现。
优化也沿用这一分工：需要改变算法工作时调整 DSL，现有关系允许更好物理实现时改进 pass。

## 与 Triton 组合

普通指针、张量广播、`tl.sum`、`tl.dot`、helper 和控制流经过共同 Triton 前端。
View 与具名轴值表达 shape identity，Encoding 表达存储映射，Level 表达归属与 carry。
Grid 继续表示应用对独立程序实例的数据划分。

建议依次阅读 [Encoding](../dsl/encoding-and-artifacts.md)、
[Level 与状态](../dsl/values-and-levels.md)和[数值操作](../dsl/functions-and-operations.md)。
完整[模型合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/programming-model/index.md)
定义作者程序与物理实现的边界。
