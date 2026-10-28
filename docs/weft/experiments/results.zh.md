# 论文结果与图片来源

本页图片采用论文已测的算子和模型数据。
按[复现流程](reproduce.md)运行对应 Bench 工作负载。

## 原生算子性能

[![算子热图与完整加速分布](../assets/paper/operators.png)](../assets/paper/operators.png)

| 平台 | 案例 | 匹配比较 | 几何平均 |
|---|---:|---:|---:|
| SG2044，RVV128 | 133 | 132 | 1.60× |
| K1／X60，RVV256 与 IME1 | 137 | 136 | 1.55× |
| K3／A100，RVV1024 与 IME2 | 136 | 135 | 1.82× |
| Xeon Platinum 8470Q，AVX-512／VNNI／AMX | 147 | 140 | 1.56× |

分布图包含全部 543 项匹配比较，其中 541 项参考 GGML，
两项 x86 BF16／INT8 dense 乘法参考 oneDNN。
倍率为相同 native workload 下的参考中位数除以候选中位数。
完整 projection 包含 activation quantization，packed leaf 采用声明的 packed-call 边界。

[553 项算子 CSV](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/paper/operators.csv)
记录 shape、数值检查、参考、耗时与 JIT 观察。

## 完整模型推理

[![九组配对配置的 prefill 与 decode](../assets/paper/models.png)](../assets/paper/models.png)

配置覆盖 Llama 2 7B Q4_0、DeepSeek 8B Q4_K_M 和 Qwen 2.5 14B Q4_K_M，
均使用八个应用线程。K3 DeepSeek 分别采用 CPU–RVV、厂商 IME2 参考策略。

| 协议 | SG2044／K1／x86 | K3 |
|---|---|---|
| Prefill | 128 tokens，初始 context 为 0 | 32 tokens，初始 context 为 0 |
| Decode | 512-token context 后生成 32 tokens | 32-token context 后生成 8 tokens |
| 执行 | GGML worker 调用 native package entry | GGML／厂商 worker 调用 native package entry |

[18 个模型阶段 CSV](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/paper/models.csv)
保存耗时、正常参考、policy、数值状态与替换计数。图中各组采用自身参考策略。

## 数据与图片

图片来自已完成的 Weft 论文 V10：

页面使用论文原始 PNG 导出；下表的 SVG 源文件保留矢量细节。

| 工具书资产 | 论文来源 | 内容 |
|---|---|---|
| `assets/paper/model.svg` | 图 4 | 逻辑作用域、共享 panel 与 partial 位宽 |
| `assets/paper/compiler.svg` | 图 5 | Canonical IR 与两个物理后端 |
| `assets/paper/operators.svg` | 图 8 | 代表角色与全部匹配算子 |
| `assets/paper/models.svg` | 图 10 | 九组模型配对配置 |

源表是 V10 的 `data/operators.csv` 和 `data/models.csv`，
Bench 公开副本保留测量值与设置。
该批数据结合目标机安装包 JIT 覆盖与受影响用例修复，各行保留选定运行记录。
V10 使用的产品／Bench 源码快照为 `369c22929`／`e17240a`。

[数据说明](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/docs/paper/README.zh-CN.md)
解释各列与参考边界。
