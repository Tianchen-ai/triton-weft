# 参考实现与计时边界

有意义的比较固定数学工作量、输入、布局、准备过程、线程与计时边界。
Bench catalog 登记这些选择。

## 选择参考

| 工作负载 | 已登记的参考族 |
|---|---|
| 量化 dot、量化／反量化、projection 与 CPU 运算 | GGML CPU/RVV 或 x86 实现 |
| Packed 矩阵调用 | 匹配的厂商 IME1／IME2 或 GGML x86 矩阵路线 |
| Dense 乘法 | GGML、显式配置的 OpenBLAS 或 oneDNN |
| 完整推理 | 使用选定 CPU／厂商后端的正常 llama.cpp |

通过 `weft-bench list --kind baselines` 和 `show CASE --target TARGET` 查看。
源码版本、构建配方和实际实现是参考身份的一部分。

## 选择对应时钟

| 测量 | 包含的工作 |
|---|---|
| Native workload | 完整原生调用，包括登记在内的算法准备 |
| Grid | 全部实例、OpenMP 启动与同步 |
| Python 热 JIT | Python 绑定、分派和正常 launcher |
| 首次 JIT | 编译、加载与首次执行 |
| 模型阶段 | llama.cpp 中实际 prefill 或 decode 窗口 |

加速比为同一计时边界下的参考中位数除以候选中位数。
目标机 JIT 完成后，Bench 还会测量其导出原生产物并与原生参考比较；
Python 热调用和首次调用分别记录。

## 固定参考与同场配对

通过 `--baseline` 复用匹配的已保存参考。
机器频率或运行状态变化时，固定参考库和版本，在同场比较。
`--sample-order alternating` 支持在相同输入上交替测量配置好的原生实现。
历史数字连同原设置保留，当前配对描述当前机器状态。

[参考与归因规格](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/development/benchmarking.md)。
