# 运行示例，学习 Triton–Weft

[English](README.md) · [工具书](https://tianchen-ai.github.io/triton-weft/zh/getting-started/examples/)

安装本项目 Triton wheel 与 NumPy，克隆仓库，并在**仓库根目录**运行下面命令。
每个文件都包含 kernel 与宿主应用；应用准备数据、启动 kernel 并打印结果。

## 1. 熟悉的 Triton

```sh
python -m examples.weft.language.triton_basics --operation affine
python -m examples.weft.language.triton_basics --operation softmax
python -m examples.weft.language.triton_basics --operation matmul
python -m examples.weft.language.triton_basics --operation scan
python -m examples.weft.language.triton_basics --operation histogram --threads 2
```

对照 [triton_basics.py](language/triton_basics.py) 阅读输出：

| 运算 | 输入与预期结果 | 重点代码 |
|---|---|---|
| Affine | 131 个 F32 → `1.25 * x - 0.5` | 三个 grid 实例与 masked tail |
| Softmax | 四行、每行 117 个 F32 → 每行和为一 | 动态行 stride、max 与 sum 归约 |
| Matmul | `[16,32] @ [32,16]` → `[16,16]` | 二维 grid、K 循环与 dot accumulator |
| Scan | 四行、每行 31 个 I32 → inclusive prefix sum | `tl.cumsum` 与末尾 mask |
| Histogram | 131 个索引、16 个 bin → 计数 9,9,9,8,… | 共享输出初始化与 `tl.atomic_add` |

## 2. 存储与复用

```sh
python -m examples.weft.language.views --size 4097
python -m examples.weft.language.encoding
python -m examples.weft.language.levels
```

- [views.py](language/views.py)：`N` 来自 typed View，最后一个 grid 实例用 subview
  覆盖最后一个元素；修改 `--size` 可观察动态 extent。
- [encoding.py](language/encoding.py)：`PackedNibbles` 声明 16 字节中的 32 个 U4，
  宿主准备高低 nibble，kernel 读取逻辑字段 `q` 并 widen 为 U8；
  第一条 record 输出 `0…15, 0…15`。
- [levels.py](language/levels.py)：row 的 stage 保存输入快照，state 累加各 block 的平方和。
  65 列对应 32、32、1 三个 block；修改 `B` 前先跟踪 `values` 与 `total` 的归属。

GGML 基础 Encoding 位于
[`triton.integrations.ggml_formats`](../../python/triton/integrations/ggml_formats.py)。
独立 Encoding 教程在消费者旁边声明自己的布局；
Bench 的 `weft_bench.kernels.ggml.layouts` 保存显式硬件布局。

## 3. 向量与矩阵算术

```sh
python -m examples.weft.language.contraction --engine vector
python -m examples.weft.language.contraction --engine auto
```

[contraction.py](language/contraction.py) 将 I8 `[16,64]` 与 I8 `[64,16]`
相乘得到 I32，并打印实际 artifact 扩展和物理 IR 阶段。
在已配置的 AMX 主机上，选择允许的 CPU 后运行：

```sh
python -m triton.tools.cpu --profile x86-amx --cpus 0 \
  examples/weft/language/contraction.py --engine matrix
```

[硬件选择](../../docs/weft/getting-started/hardware.zh.md)说明 K1／K3 注册、
核心分组与 contraction 路线。

## 4. Autotune、导出与原生 C++

```sh
python -m examples.weft.deployment.autotune_export /tmp/tuned-affine
python -m examples.weft.language.views --size 64 --threads 1 --output /tmp/affine-native
python -c "from triton.runtime.package import sdk_prefix; print(sdk_prefix())"
cmake -S examples/weft/deployment -B /tmp/weft-native-build \
  -DWeft_DIR=/printed/sdk/path/lib/cmake/Weft
cmake --build /tmp/weft-native-build
/tmp/weft-native-build/weft-native-call /tmp/affine-native
```

Artifact 使用新目录。[autotune_export.py](deployment/autotune_export.py)
在 B=32／64／128 中选择、导出并重新加载调用。
[native_call.cpp](deployment/native_call.cpp) 加载 64 元素的 View 程序，
绑定调用者存储，输出 `-0.5 0.75 2 3.25`。

## 5. 框架与模型

| 示例 | 环境与下一步 |
|---|---|
| [pytorch_cpu.py](frameworks/pytorch_cpu.py) | 安装受支持 CPU PyTorch，执行 `python -m triton.tools.pytorch`，再运行 `python -m examples.weft.frameworks.pytorch_cpu`，学习 eager Tensor 调用与 Inductor normalization |
| [llama_cpp.py](deployment/llama_cpp.py) | `python -m examples.weft.deployment.llama_cpp /tmp/q8-package --profile native` 导出 Q8_0 row-dot；按[部署教程](../../docs/weft/integrations/llama-cpp.zh.md)构建并绑定 GGML |

完整优化程序、数值检查和性能比较使用
[Triton–Weft Bench](https://github.com/Tianchen-ai/triton-weft-bench)。
[复现指南](../../docs/weft/experiments/reproduce.zh.md)连接语言示例与论文算子、模型工作负载。
