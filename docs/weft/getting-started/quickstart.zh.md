# 第一个 Weft kernel

这个完整的逐行计算组合带类型的存储、共享输入与有序状态更新。
完成[安装](installation.md)后，运行 `python -m examples.weft.quickstart`。

## 编写并调用程序

将下列程序保存为 `row_sum_squares.py`，运行 `python row_sum_squares.py`：

```python
from __future__ import annotations
import numpy as np
import triton
import triton.language as tl

@triton.jit(model="weft")
def row_sum_squares(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M,)],
                    BLOCK: tl.constexpr = 32):
    with tl.level.rows(M, group=1) as row:
        values = tl.stage(tl.load(X[row]))
        total = tl.state(tl.f32, [1], init=0)
        with tl.level.blocks(N, extent=BLOCK) as block:
            part = values[:, block]
            total += tl.reduce(part * part, axis="n")
        tl.store(Y[row], total)

x = np.random.default_rng(19).normal(size=(4, 65)).astype(np.float32)
y = np.empty(4, dtype=np.float32)
compiled = row_sum_squares[(1,)](x, y, BLOCK=32, num_cpu_threads=1)
np.testing.assert_allclose(
    y, np.sum(x.astype(np.float64) ** 2, axis=1), rtol=1e-5, atol=1e-5)
print(y, list(compiled.asm))
```

程序将每个输出与双精度参考比较。

## 顺着源程序理解关系

View 类型根据输入 shape 绑定 `M` 与 `N`。行层拥有一份 staged 输入观察值
和一个初始化的状态。内层 block 分别处理 32、32、1 个有效元素，更新同一状态。
最终 store 读取完整更新链。改变 `BLOCK` 会特化逻辑分组。

编译器利用这些关系构造目标向量和局部存储。
`num_cpu_threads` 选择 CPU 工作线程数，grid 表示独立程序实例。
类型与绑定相同的后续调用复用编译结果。

## 查看 IR 与导出

仓库示例还能保存真实编译 IR，并导出原生产物：

```sh
python -m examples.weft.quickstart \
  --dump-ir /tmp/weft-quickstart-ir --export /tmp/weft-quickstart-native
```

`compiled.asm` 提供 Canonical `weftir`、目标 `x86ir` 或 `riscvir`，
以及 `llir` 或 `c`。导出保存原生库、ABI 与依赖，
[原生 SDK](../compiler/native-sdk.md)继续介绍加载和 C/C++ 调用。

接下来阅读 [Encoding](../dsl/encoding-and-artifacts.md)、
[Level 与状态](../dsl/values-and-levels.md)，或运行
[完整投影、注意力、Welford 与 GEMM](examples.md)。
