# 运算、数值位宽与 helper

标准 Triton 运算保留原有语言入口；Weft 在需要时进一步显式表达逻辑轴与数值实现。

## 熟悉的张量运算

| 运算 | 用途 |
|---|---|
| `tl.arange`、索引与广播 | 构造逻辑 tile 和指针偏移 |
| `tl.load/store(..., mask=...)` | 访问有效内存元素 |
| `tl.sum/max`、`tl.reduce` | 归约张量轴；tuple combiner 可携带多个结果 |
| `tl.dot(a, b, acc)` | `[..., M, K]` 与 `[..., K, N]` 相乘并累加 acc |
| `tl.cumsum`、`tl.associative_scan` | 表达前缀计算 |
| `tl.atomic_add` 等指针 atomic | 通过显式原子操作更新共享存储 |
| `to/cast/bitcast` | 使用 Triton 的 typed conversion 规则 |

[基础示例](../getting-started/examples.md)覆盖 masked tail、strided row、
dot accumulator、scan 与 histogram。

## 显式有限位宽算术

```python
partials = tl.mac_groups(weights, activation, n=4, into=tl.i16)
wide = tl.widen(partials, tl.i32)
total = tl.reduce(wide, axis="k", op="add")
```

分组与 I16 partial 都属于数值程序，最终归约前先 widen。
编译器通过共同整数范围分析，证明哪些机器指令能实现这些准确边界。

`tl.narrow` 保留显式转换位置及其舍入／饱和合同。
整数与 packed field 遵循精确合同，浮点遵循声明的舍入与误差容差。
编译器可在已确认的 FMA 合同内融合局部、单消费者的浮点乘加。

## 具名轴 contraction

```python
dot = tl.reduce_dot(w.q, x.q, over="k", acc_dtype=tl.i32, engine="vector")
```

具名轴 Value 按 axis identity 对齐，归约消去指定轴并保留其余轴。
普通矩阵 tile 使用 `tl.dot(a, b, acc))，继续采用位置轴矩阵约定。
两种形式都接受逐 contraction 的 [engine 约束](../getting-started/hardware.md)。

## 函数与 source 参数

普通 `@triton.jit` helper 默认随调用者模型编译；也可显式选择模型。
helper 通过共同前端组合 typed Value、Encoding 字段、Level 作用域和 effects。

`tl.constexpr` 与声明的有限 source 参数用于选择作者程序，例如 block extent、
group 大小和布局实例。`triton.Config.kwargs` 绑定这些参数，
`backend_options` 则控制同一程序的物理选择。
[Autotune 教程](../compiler/runtime.md)展示两者如何放入同一工作流。

[运算完整合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/dsl/functions-and-operations.md)
详细规定 signedness、NaN／signed-zero、归约 identity、widen、narrow 和 helper 语义。
