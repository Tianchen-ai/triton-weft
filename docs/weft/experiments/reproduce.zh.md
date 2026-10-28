# 复现算子与模型实验

使用 **Triton–Weft 编译作者程序**，使用 **Triton–Weft Bench 准备、检查并测量工作负载**。
Bench 包含可安装 kernel corpus、目标 catalog、参考库配方和模型协议。

## 1. 安装与配置

在各执行机器安装 Triton–Weft，再安装 Bench：

```sh
git clone https://github.com/Tianchen-ai/triton-weft-bench.git
cd triton-weft-bench
python -m pip install '.[verification]'
mkdir -p .local
cp configs/example.json .local/bench.local.json
export WEFT_BENCH_CONFIG="$PWD/.local/bench.local.json"
```

填写目标 Python、允许的 CPU 集合、原生 SDK、参考库路径和输出目录，
选择 `x86-1t`、`sg2044`、`k1` 或 `k3-a100` catalog。
SSH 目标按 [Bench 配置指南](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/targets.md)
填写连接与部署字段；参考库按
[登记配方](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/src/weft_bench/catalogs/references.json)构建。

目标机 JIT 使用 `weft_bench.adapters.triton_jit:TritonJitAdapter`
和目标已安装的 Python。选核及 K3 注册先于能力发现，本机 x86 沿用相同流程。

## 2. 查看选定程序

```sh
weft-bench list --kind targets
weft-bench list --kind suites --target k1
weft-bench list --kind baselines
weft-bench show q4_k_dot --target k1
```

`show` 解析 DSL 文件／symbol、source 参数、target、物理选项、参考与计时范围。
基础 Encoding 位于 `triton.integrations.ggml_formats`，
显式候选布局位于 `weft_bench.kernels.ggml.layouts`。

## 3. 运行匹配的算子比较

配置已有参考后，运行选定候选：

```sh
weft-bench op q4_k_dot --target k1 --output output/current/k1/q4_k_dot
```

为该用例建立并复用 GGML 参考：

```sh
weft-bench op q4_k_dot --target k1 --impl ggml \
  --output output/baselines/k1/q4_k_dot
weft-bench op q4_k_dot --target k1 \
  --baseline output/baselines/k1/q4_k_dot \
  --output output/current/k1/q4_k_dot
```

Bench 为双方提供相同输入与数值 oracle。JIT adapter 在目标机执行 `kernel[grid](...)`，
记录首次调用、缓存命中与热调用，并通过这次成功 JIT 产物的导出入口，
测量与 GGML 边界匹配的原生计算。

## 4. 运行完整登记套件

配置机器和匹配参考后执行：

```sh
weft-bench op --suite full --target sg2044  --output output/current/sg2044/full
weft-bench op --suite full --target k1      --output output/current/k1/full
weft-bench op --suite full --target k3-a100 --output output/current/k3-a100/full
weft-bench op --suite full --target x86-1t  --output output/current/x86-1t/full
```

| Catalog | 算子—平台案例 |
|---|---:|
| SG2044 | 133 |
| K1 | 137 |
| K3 A100 | 136 |
| x86 单线程 | 147 |
| 合计 | 553 |

每项保留选定 shape、作者程序和配置。日常开发可只选择受影响用例。

## 5. 准备模型 package

配置正常／集成版 llama.cpp 和本地 GGUF 文件，查看模型协议与 package 配方：

```sh
weft-bench list --kind models --target k1
weft-bench list --kind replacements
weft-bench prepare-model --target k1 --model llama2-7b-q4_0 \
  --output output/current/k1/llama2-package
```

准备结果给出可部署 package 路径。将所选模型配置的
`implementations.weft.package` 设为执行机器上的该路径，
package 在推理前固定角色与 kernel 绑定。
[llama.cpp 教程](../integrations/llama-cpp.md)说明原生 adapter。

## 6. 测量 prefill 与 decode

```sh
weft-bench e2e llama2-7b-q4_0 --target k1 --impl normal --phase both \
  --output output/baselines/k1/llama2
weft-bench e2e llama2-7b-q4_0 --target k1 --impl weft --phase both \
  --baseline output/baselines/k1/llama2 \
  --validate --validation-reference output/baselines/k1/llama2-logits \
  --output output/current/k1/llama2
```

采用 catalog 的输入长度、线程和重复次数。`--phase` 选择测量窗口；
窗口内计数说明实际调用了哪些绑定函数，登记的 logits 检查在计时外执行。

## 查看结果

每次运行生成 `results.json` 和表格，保存数值状态、耗时、参考、配置与复现信息。
Native、grid、Python 与模型使用各自[计时边界](../development/benchmarking.md)。
当前结果放在 `output/current`，固定参考放在 `output/baselines`。

[论文结果](results.md)提供图片、CSV 和测量设置。
