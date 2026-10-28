# 从物理操作生成原生代码

Emission 将验证后的物理决定翻译成目标工具链输入。
生成代码通过 `CompiledKernel.asm` 和导出产物提供查看入口。

## RISC-V

RVV 操作翻译为 typed intrinsic，已选扩展操作可使用 typed local assembly。
布局、active length、conversion、局部 packing、流水版本和资源决定均已物化。
Clang 23 按指定 ISA、ABI 与 VLEN 编译生成的 C。

先查看 `asm["riscvir"]`，再查看 `asm["c"]`。
对 load 跟踪逻辑 region、物理访问和有效域 mask；
对 contraction 跟踪所选局部操作和 accumulator 表示。

## x86

x86 管线将物理 vector/memory 程序降低到 LLVM dialect，再翻译为 LLVM IR。
LLVM x86 codegen 按目标 feature set 完成最终机器寄存器与指令选择。

先查看 `asm["x86ir"]`，再查看 `asm["llir"]`。
SIMD 宽度与 AMX 使用体现在已选操作和 artifact 要求中。

## 原生产物

两条路线都生成动态库、typed ABI metadata 和运行依赖。
Triton 可直接加载执行 JIT，也可导出给 [C++](native-sdk.md) 与
[llama.cpp](../integrations/llama-cpp.md)。

[完整 emission 规格](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/emission.md)。
