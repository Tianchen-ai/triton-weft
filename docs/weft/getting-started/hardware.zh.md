# 目标、执行核心与 engine

明确选择三个相关事项：**进程在哪些核心执行**、**编译可以使用哪些指令**，
以及**当前 contraction 要求哪条路线**。

## 发现并选择核心

```sh
python -m triton.tools.cpu --list
python -m triton.tools.cpu --profile k3-a100 --cpus 8-15 application.py
python -m triton.tools.cpu --profile k3-x100 --cpus 0-7 application.py
```

K3 A100 启动器先通过 `/proc/set_ai_thread` 注册 PID，再设置 CPU affinity，
最后启动应用。目标发现基于选定的核心组。
K3 X100 的 VLEN 为 256，A100 为 1024，应选择同质核心集合。

## 构造编译目标

```python
from triton import cpu

print(cpu.devices())
print(cpu.target())          # 本机能力发现
print(cpu.target("k3-a100")) # 显式编译 profile
```

| Profile | 架构／向量能力 | 矩阵能力 |
|---|---|---|
| `sg2044` | RVV，VLEN 128 | — |
| `k1` | RVV，VLEN 256 | IME1 |
| `k3-x100` | RVV，VLEN 256 | — |
| `k3-a100` | RVV，VLEN 1024 | IME2 |
| `x86-avx2` | x86-64-v3，256-bit SIMD | — |
| `x86-avx512` | x86-64-v4，512-bit SIMD | — |
| `x86-amx` | x86-64-v4，512-bit SIMD | AMX INT8／BF16 |

AOT 时把目标传给 `triton.compile(ASTSource(...), target=...)`。
Profile 展开为能力事实；`CPUTarget` 也接受显式 ISA、ABI、向量长度和扩展字段。

## 约束一个 contraction

```python
tl.dot(a, b, acc)                    # 合法矩阵优先，其次向量
tl.dot(a, b, acc, engine="matrix")   # 要求合法矩阵实现
tl.dot(a, b, acc, engine="vector")   # 从向量实现中选择
```

默认规则保持源 dtype 与 accumulator 合同。向量实现包括 RVV、SIMD 和 VNNI。
同一 kernel 可以组合向量 decode、矩阵算术和向量后处理；
目标 passes 在每个操作的路线内选择布局与资源。

通过 `compiled.metadata.kernel_abi["used_extensions"]` 和物理 IR 查看实际要求。
Artifact requirements 记录执行能力；条件矩阵路径的实际调用由 Bench observer 单独观察。

Grid 大小与 `num_cpu_threads` 相互独立。Runtime 检查实际 worker，
为要求 AMX 的 artifact 处理相应执行权限。
