# 选择计算路线、目标与执行核心

`engine` 约束一个 contraction。编译目标说明允许生成的指令；CPU 集合说明程序实际在哪些
核心执行。三者分别表达作者程序、目标能力和进程部署。

```python
tl.dot(a, b, acc)                    # auto：合法矩阵优先，否则向量
tl.dot(a, b, acc, engine="matrix")  # 要求矩阵实现；不支持时报出原因
tl.dot(a, b, acc, engine="vector")  # RVV/SIMD/VNNI，不使用矩阵指令
tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32, engine="vector")
```

同一 kernel 可以组合向量解码、矩阵乘加和向量后处理。编译器不在向量与矩阵之间做成本竞选；
路线内的 packet、LMUL、fragment、布局、流水、活跃值和寄存器资源仍由物理 passes 处理。
选择矩阵不会把 F32 自动转成 BF16，也不会改变累加器位宽。显式矩阵要求完整数值合同能由
矩阵路径完成；如当前 BF16 路径需要按输入数值回到 SIMD，显式矩阵会拒绝该配置。

默认路线可以包含合法性分支。产物含 AMX 或 IME 指令，意味着需要该执行能力，不意味着
某次调用一定经过了条件分支中的矩阵指令。Bench 分开展示产物要求与实际观察范围。

## 编译目标

```python
from triton import cpu
from triton.compiler import ASTSource

print(cpu.profiles())
target = cpu.target("k3-a100")
compiled = triton.compile(ASTSource(kernel, signature, constants), target=target,
                          options={"toolchain": cross_toolchain})
compiled.export("native-kernel")
```

| 配置 | 架构与向量能力 | 允许的矩阵扩展 |
|---|---|---|
| sg2044 | RISC-V RVV，VLEN128 | 无 |
| k1 | RISC-V RVV，VLEN256 | IME1 |
| k3-x100 | RISC-V RVV，VLEN256 | 无 |
| k3-a100 | RISC-V RVV，VLEN1024 | IME2 |
| x86-avx2 | x86-64-v3，256-bit SIMD | 无 |
| x86-avx512 | x86-64-v4，512-bit SIMD | 无 |
| x86-amx | x86-64-v4，512-bit SIMD | AMX INT8/BF16 |

命名配置是可读的目标参数集合，不按机器名称分派数学 kernel。需要更多 ISA 特征时，可直接
构造 `CPUTarget`。目标能力可以扩展；不同目标仍消费同一 Canonical 程序。

## 执行核心

```sh
python -m triton.tools.cpu --list
python -m triton.tools.cpu --profile k3-a100 --cpus 8-15 application.py
python -m triton.tools.cpu --profile k3-x100 --cpus 0-7 application.py
```

`devices()` 在隔离进程中探测核心，返回同质 CPU 组及启动要求。K3 A100 启动器先将当前 PID
登记到 `/proc/set_ai_thread`，再设置 affinity，最后启动应用。混合 VLEN 或缺少所需 ISA 的
集合会在启动应用前被拒绝。JIT 保留启动器选择的 profile：指定 AVX2 不会自动升级成 AMX。

grid 数量与 `num_cpu_threads` 分开。多实例通过 OpenMP 启动；已有 GGML worker 使用导出的
单实例入口。runtime 检查实际 worker 的能力；资源分析已经在编译期完成，不在推理时重算。
AMX tile-state 权限仅由实际要求 AMX 的产物申请。
