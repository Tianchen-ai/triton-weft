# 可运行示例

示例是包含明确数据准备和输出的小型应用。安装 Triton–Weft 与 NumPy 后，
从仓库根目录运行；可同时打开
[源码索引](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/README.zh-CN.md)。

## 1. 普通 Triton

```sh
python -m examples.weft.language.triton_basics --operation affine
python -m examples.weft.language.triton_basics --operation softmax
python -m examples.weft.language.triton_basics --operation matmul
python -m examples.weft.language.triton_basics --operation scan
python -m examples.weft.language.triton_basics --operation histogram --threads 2
```

| 示例 | 阅读重点 | 预期行为 |
|---|---|---|
| Affine | program_id、arange 和尾部 mask | 131 个输出等于 `1.25*x-0.5` |
| Softmax | 动态行 stride 与 max/sum | 四行各 117 个元素，逐行归一化 |
| Matmul | 二维 grid 与 K 累加 | `16×32 @ 32×16` 得到 `16×16` |
| Scan | Inclusive `tl.cumsum` | 对每行 31 个 I32 求前缀和 |
| Histogram | 共享计数的 atomic 更新 | 将 131 个观测分入 16 个 bin |

## 2. 加入显式关系

```sh
python -m examples.weft.language.views --size 4097
python -m examples.weft.language.encoding
python -m examples.weft.language.levels
python -m examples.weft.language.contraction --engine vector
```

View 表达动态 extent 与子区域；Encoding 连接 packed bytes 与逻辑字段；
Level 连接 staged row、accumulator 和 block update；
contraction 打印所选 artifact 的实际扩展。
[示例解读](../dsl/examples.md)逐一解释这些决定。

## 3. 调优与部署

```sh
python -m examples.weft.deployment.autotune_export /tmp/tuned-affine
```

程序调优 source block 大小，导出选中的 kernel，再重新加载执行。
随后可学习完整 [C++ 应用](../compiler/native-sdk.md)或
[llama.cpp 接入](../integrations/llama-cpp.md)。

## 4. CPU Tensor 与图编译

```sh
python -m triton.tools.pytorch
python -m examples.weft.frameworks.pytorch_cpu
```

在受支持 PyTorch 环境中，示例先对 CPU Tensor 调用手写 kernel，
再通过 Inductor 编译 normalization 图。详见[框架配置与流程](../integrations/overview.md)。

Bench 为这些相同应用和完整优化 corpus 提供独立数值检查。
按[实验复现](../experiments/reproduce.md)比较 kernel 和完整推理。
