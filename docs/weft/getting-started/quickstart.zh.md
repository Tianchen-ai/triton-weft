# 第一个 CPU kernel

完成[安装](installation.md)后，将下面程序保存为 `affine.py`。
应用持有输入输出数组，kernel 计算 `Y = scale * X + bias`。

```python
import numpy as np
import triton
import triton.language as tl

@triton.jit(model="weft")
def affine(X, Y, N, scale, bias, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    active = offsets < N
    x = tl.load(X + offsets, mask=active, other=0.0)
    tl.store(Y + offsets, x * scale + bias, mask=active)

x = np.arange(131, dtype=np.float32)
y = np.empty_like(x)
grid = (triton.cdiv(x.size, 64),)
compiled = affine[grid](x, y, x.size, 1.25, -0.5, B=64, num_cpu_threads=2)
print(y[:4])
print("Compilation stages:", list(compiled.asm))
```

执行 `python affine.py`，前四个输出为 `[-0.5, 0.75, 2.0, 3.25]`。

## 理解启动方式

`B` 是编译期 block 大小；`N`、`scale` 和 `bias` 是运行时参数。
每个 `program_id` 选择 64 个逻辑元素，三个实例覆盖 131 个元素，
最后一个实例的 mask 控制实际访存。`num_cpu_threads=2` 独立设置 OpenMP 线程数。

再次调用同一 kernel 时复用编译好的特化。改变 `B` 这样的 constexpr 会选择另一份特化；
改变数组内容则复用相同程序。

## 查看并导出

```python
print(compiled.asm["weftir"])
compiled.export("/tmp/affine-artifact")  # 使用一个新目录
loaded = triton.load("/tmp/affine-artifact")
loaded[grid](x, y, x.size, 1.25, -0.5)
```

导出的产物固定 `B=64`，加载后的 callable 接收其余运行时参数。
RISC-V 可查看 `riscvir` 和 `c`，x86 可查看 `x86ir` 和 `llir`，两者都产生 `so`。

继续阅读[示例](examples.md)，学习行 softmax 和矩阵乘；或阅读
[Encoding](../dsl/encoding-and-artifacts.md)，学习打包存储。
