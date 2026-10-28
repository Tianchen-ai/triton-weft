# Encoding、View 与持久数据

Encoding 把逻辑字段坐标映射到存储字节与 bit；字段的数值意义由 kernel 中的算术表达。
View 则将这一存储合同与逻辑 shape、访问关系组合起来。

## 声明打包 record

可运行的 [Encoding 示例](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/language/encoding.py)
用 16 个字节存放 32 个无符号四位值：

```python
@triton.encoding
class PackedNibbles:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 32
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)
```

字段坐标 `i` 对应字节 `i % 16`。坐标 0–15 使用低 nibble，16–31 使用高 nibble。
`elements` 表示每条 record 的逻辑元素数，字段映射决定实际字节跨度。

宿主显式准备这些字节：

```python
logical = (np.arange(64, dtype=np.uint8) % 16).reshape(2, 32)
storage = logical[:, :16] | (logical[:, 16:] << 4)
x = tl.View(storage, shape=(logical.size,), encoding="PackedNibbles")
```

kernel 中的 `tl.load(X[block]).q` 得到字段的逻辑 Value；
`tl.widen(record.q, tl.u8)` 保持坐标并提升数值位宽。

## 选择布局声明

| 构造 | 含义 |
|---|---|
| Dense `tl.View[tl.f32, (M, N)]` | 带逻辑 extent 的 typed row-major 存储 |
| `grouped(elements=N)` | 在每组 N 个元素内独立重复存储映射 |
| `bit_layers(elements=P, order=...)` | 将不同 P 元素层的对应值放入同一字节的 bit range |
| `pack_fields(group=..., fields=..., low_bits=..., order=...)` | 多个连续字段共用紧凑字节容器 |
| `alignment` 和显式 padding | record 起点对齐，以及没有逻辑字段的存储区间 |

使用 GGML 格式时，按模块实际职责导入：

```python
from triton.integrations import ggml_formats
# 参数类型示例：tl.View[ggml_formats.Q4_K, (N, K)]
```

该模块保存基础存储声明。Bench 的 `weft_bench.kernels.ggml.layouts`
保存显式候选布局与 builder，按 IME1、IME2、AMX 和行排列组织。

## View、切片与 alias

View slice 选择内存 region；`tl.load` 保存程序执行到该处时读到的值。
非负 dense stride 通过 `@triton.jit(model="weft", strided=("X",))` 声明。
只读 zero-stride 可用于广播，可写 View 的元素存储应互不重叠。

`alias_groups={"X": "input", "Y": "output"}` 声明不同存储组。
同组参数允许重叠，不同组是调用方的互不重叠承诺；未声明参数进入共同 may-alias 组。

## 持久布局与局部 packing

`@triton.derive` 描述显式持久布局族。例如 `tl.interleave(W, rows=16)`
生成 16 条基础 record 的 byte-major cohort；准备与消费共享确定的布局身份。
初始化成本和存储由应用管理。

已加载 Value 的局部 packing 是其生命周期内的编译器表示选择，
物理程序记录分配、转换、使用和释放。这使同一源布局能在 RVV、IME、AMX 上采用不同局部排列。

[完整 Encoding 与存储合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/dsl/encoding-and-artifacts.md)
给出字段坐标公式、派生布局、padding、尾块及固定 ABI 边界。
