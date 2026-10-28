# CPU Tensor、PyTorch 与组合

CPU Tensor 可以传入与 NumPy 相同的 Triton kernel。
Runtime 借用其存储，保留 dtype、shape、stride、offset 和 alias。

## Eager Tensor 调用

```python
import torch
import triton
from examples.weft.language.triton_basics import affine

x = torch.linspace(-1, 1, 131, device="cpu")
y = torch.empty_like(x)
affine[(triton.cdiv(x.numel(), 64),)](
    x, y, x.numel(), 1.25, -0.5, B=64, num_cpu_threads=2
)
```

应用分配输出并选择 grid；读写 effects 与存储关系来自编译使用的同一 Canonical interface。

## Inductor 自动生成 kernel

在受支持的 PyTorch 2.14 环境中安装一次源码接缝：

```sh
python -m triton.tools.pytorch
python -m examples.weft.frameworks.pytorch_cpu
```

示例编译下面的计算图：

```python
def normalize(value):
    centered = value - value.mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(
        (centered * centered).mean(dim=-1, keepdim=True) + 1e-5
    )

compiled = torch.compile(normalize, backend="inductor", fullgraph=True,
                         options={"cpu_backend": "triton"})
```

Inductor 提供图分解、融合与 Triton 生成，
生成 kernel 经过共同前端和 CPU 编译路径。
接缝保留模型身份、effects、FakeTensor metadata 与启动行为。

## 状态与梯度

宿主持有模型 state、workspace 和准备过程。
多个 kernel 协作时显式排列调用顺序；
backward kernel 通过框架 autograd 接口注册，
corpus 的 `weft_bench.kernels.dense.torch_affine` 提供例子。

继续阅读[模型组合](../dsl/programs.md)、[原生 SDK](../compiler/native-sdk.md)
或 [llama.cpp 部署](llama-cpp.md)。
