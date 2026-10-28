# Triton 内部的 Weft 编译核心

本目录包含 Canonical Kernel IR、共享分析与变换、RISC-V RVV/IME 与 x86 SIMD/AMX
物理编译，以及 C/C++ native runtime。Python 语言、JIT、autotune、backend 与 Triton 一起构建安装。

公共入口是 `@triton.jit(model="weft")` 和 `kernel[grid](...)`，没有独立 Weft Python 发行包或 CLI。

`lib/Dialect/Kernel` 保存作者语义，`lib/Analysis` 与 `lib/Transforms` 保存共同机制。
`lib/Target/{RISCV,X86}` 均按 Backend、Platform、Analysis、Conversion、Transforms、Emission
组织；RISC-V 的自定义物理方言与 x86 复用的 MLIR 方言有不同表示，见[源码地图](../../docs/weft/development/upstream.zh.md)。

从[工具书](../../docs/weft/index.zh.md)、[安装说明](../../docs/weft/getting-started/installation.zh.md)和
[native SDK](../../docs/weft/compiler/native-sdk.zh.md)开始。Weft 原代码保留 [Apache-2.0](LICENSE) 归属。
