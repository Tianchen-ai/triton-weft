# JIT、缓存与 autotune

Weft kernel 使用 Triton 的 `JITFunction`、特化、缓存、`CompiledKernel` 和 autotuner。
CPU backend 提供目标发现、原生编译和 grid launcher。

## 首次与后续调用

```python
compiled = kernel[(instances,)](X, Y, B=64, num_cpu_threads=2)
kernel[(instances,)](X, Y, B=64, num_cpu_threads=2)
```

首次调用解析目标与参数合同，特化源码、编译并加载原生程序；第二次复用该特化。
依赖包括 helper、Encoding／derived 声明和源常量。
运行时 shape、stride 和 scalar 通过 typed ABI 传递。

通过 `TRITON_CACHE_DIR` 指定应用编译缓存。
工具链、目标 feature、ABI、编译和链接参数进入编译身份；
显式 sysroot 使用应用声明的 `sysroot_identity`。

## Grid 与线程

Grid 指定独立实例，`num_cpu_threads` 指定 OpenMP 执行。
单实例直接调用原生代码，多实例单线程使用原生循环，多线程使用原生 OpenMP launcher。
`num_cpu_threads=0` 采用 OpenMP 当前设置。

应用通过 `tl.program_id` 划分数据，每个实例运行一份有序源程序。
已有 worker 的框架可以使用导出的单实例入口，详见[原生 SDK](native-sdk.md)。

## 选择 source 与物理配置

```python
configs = [
    triton.Config({"B": 32}, num_cpu_threads=2),
    triton.Config({"B": 64}, num_cpu_threads=2),
    triton.Config({"B": 128}, num_cpu_threads=2),
]
tuned = triton.autotune(configs=configs, key=["N"])(affine)
compiled = tuned[lambda meta: (triton.cdiv(n, meta["B"]),)](
    x, y, n, 1.25, -0.5
)
print(tuned.best_config)
```

`kwargs` 绑定 source 参数；`backend_options` 绑定
`pipeline_depth`（1–4）等物理参数；`num_cpu_threads` 配置 launcher。
Triton 提供候选枚举、pruning、hook、计时和选择缓存；
候选均经过数值、effect 与资源合法性检查。

Autotune 在保持 alias 关系的私有存储中试跑，每轮恢复可变输入，
随后在调用者存储上执行一次选中的 kernel。

## CPU buffer

NumPy buffer 和 CPU Torch Tensor 借用原有存储。
View metadata 保留 shape、声明的非负 stride、offset 与 alias。
Packed storage 使用 `tl.View(bytes, encoding=..., shape=...)`；
调用方在执行期间保持存储有效。

## 导出与重新加载

```python
compiled.export("/tmp/selected-kernel")
loaded = triton.load("/tmp/selected-kernel")
loaded[(instances,)](X, Y)
```

按导出 kernel 的运行时参数调用，constexpr 已经固定。
导出使用新目录，其中包含原生库、header、manifest、Triton metadata 和运行依赖。
参见[完整导出示例](../getting-started/examples.md)。

[详细 runtime 合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/runtime.md)。
