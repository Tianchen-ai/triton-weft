# 模型程序与宿主组合

可安装的 `weft_bench.kernels` 包含完整作者程序，
与教程使用相同 Triton 语言、编译器和 runtime。

| 程序族 | 源码表达的计算 |
|---|---|
| Normalization | RMSNorm、中心方差 LayerNorm、residual 融合和显式行／tile 变体 |
| Activation | SiLU、SwiGLU、softplus、add、copy、fill 与选择 |
| Linear | Dense contraction、bias／activation epilogue 和嵌套 Level |
| Attention | Split／interleaved RoPE、KV 写入与 online causal grouped-query attention |
| MoE | Router top-k、token 分组、ragged expert projection 和加权合并 |
| Mamba | Causal depthwise convolution 与 selective state recurrence |

## 调用程序

```python
from weft_bench.kernels.programs.normalization import rms_norm
rms_norm[(1,)](X, Weight, Y, epsilon=1e-5, BLOCK=32)
```

该程序包含完整行遍历：X／Y 是二维 View，Weight 是一维 View，
BLOCK 是显式 source 参数，一个实例覆盖源码中的遍历。

## 组合宿主操作

```python
from weft_bench.kernels.programs import get, Operators

config = get("rms_norm").config(BLOCK=64)
with Operators(configs={"rms_norm": config}) as operators:
    result = operators.rms_norm(x, weight)
```

宿主组合分配输出、检查应用 shape，并按源码顺序调用 kernel。
配置使用 Triton Config，调优候选交给同一 Triton autotuner。

应用持有 KV cache、workspace、prepared weight 与递推状态，
决定准备何时执行、哪些调用共享状态。训练通过显式 backward kernel 或框架注册表达。

[Bench catalog](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/catalogs.md)
提供 source symbol、所选布局和登记工作负载。
