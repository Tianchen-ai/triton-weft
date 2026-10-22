# C/C++ SDK 与 llama.cpp 接入

## 从同一个 CompiledKernel 导出

Triton 的 CPU backend 直接生成 native library。`CompiledKernel.export` 发布该库、typed header、
ABI/目标 metadata 和运行依赖；`triton.load` 读取同一产物，不需要原 DSL 或现场编译器。

```python
import triton
from triton.integrations.llama_cpp import create_package
compiled_dot.export("/application/kernels/dot")
package = create_package(
    "/application/kernels/model",
    {"projection_dot": compiled_dot},
    bindings={"ggml.vec_dot": {"q4_0": "projection_dot"}},
)
```

通用 Package 接收已经选定的 CompiledKernel，包括 triton.load 返回的对象。
它不编译、不调优、不生成算法或 workspace；绑定是应用明确选择的模型角色。
所有 entries 必须使用共同 ABI/VLEN；可选的 llama.cpp integration 额外核对 GGML
角色的 Encoding、shape、stride 和已用扩展。通用 Package 不导入 GGML 校验器。
包内 kernel 路径相对 package.json，发布前完成全部 entries，已有目录拒绝覆盖。

一个 kernel.so 包含两个入口：manifest.json 顶层描述单实例函数和 typed dispatch；
其 grid 字段描述原始程序 ABI 和 OpenMP launcher。两者来自同一数值程序。
llama.cpp 使用单实例入口，不再创建线程团队；显式 grid 调用使用原始 entry，
不能把单实例 entry pointer 传入 grid launcher。

C/C++ SDK 随 Triton wheel 分发，路径可由以下入口取得：

```python
from triton.runtime.package import sdk_prefix
print(sdk_prefix())
```

SDK 包含 include/weft/runtime.h、runtime.hpp、lib/libweft_runtime.so、
lib/cmake/Weft/WeftConfig.cmake 和 share/llama.cpp。Weft 是内部 native ABI 名称，
不要求安装另一份 Python 包。

```cmake
find_package(Weft CONFIG REQUIRED)
add_executable(application application.cpp)
target_compile_features(application PRIVATE cxx_std_17)
target_link_libraries(application PRIVATE Weft::Runtime)
```

Weft_DIR 指向 SDK 的 lib/cmake/Weft。运行时依赖与目标 ABI 必须匹配。
SDK 共用编译器的 ISA/capability 解释，不链接 MLIR 编译管线，也不调用 Python。
数学运算全部来自作者程序：RISC-V 经 Physical IR 生成 intrinsic C/local asm，
x86 经 Physical IR 生成 LLVM IR。两者共用原生 ABI。

## 调用、存储与生命周期

C 入口返回 `weft_status` 与可选 `weft_error`，错误消息包含参数或执行能力不匹配的原因。
所有新建 handle 的输出在失败时为 null。C++ wrapper 将相同错误转为 `weft::error`，并提供
move-only RAII。by-value f16/bf16 没有通用 C scalar 表示时明确拒绝；dense/packed View 的
Encoding 与字节表示仍由 compiler ABI 决定。

| 对象 | 所有权与调用合同 |
|---|---|
| context | 在当前线程 affinity 上探测目标；kernel 持有其执行能力信息 |
| kernel | 保持 native library、入口和 immutable ABI；释放外层 handle 不使已有 call 失效 |
| View | 借用调用者存储；容量、shape、stride、Encoding、可写性和 alias 合同在 prepare 时检查 |
| buffer | SDK 显式分配的对齐字节存储；不自动量化、初始化、repack 或清零状态 |
| bindings | 按参数名保存 metadata 和 scalar 的副本；`retain` 使 builder/call 持有 SDK buffer |
| call | 保存完整 ABI 参数；prepare 后外层描述符可释放，借用的数据字节仍须保持存活 |
| sequence | 持有一列 prepared calls，按作者指定顺序同步执行；没有隐式 worker、融合或重排 |

`kernel.storage_layout(argument, dimensions, strides)` 解析 dense/packed record、row cohort、
alignment 与具体 View span；调用者可以 `allocate()`，也可 `borrow()` 现有容量。
strides 是非负 element strides；未声明 strided 的 View 要求自然连续布局。
可写自重叠、非完整 record/cohort、signed pointer 范围外的 span 均被拒绝。空 View 不访问存储。
allocation span 和 prepare 使用同一个 `Layout` 合同计算。

```cpp
weft::context device;
weft::package package(device, package_directory);
auto kernel = package.get_kernel("affine");
auto output_layout = kernel.storage_layout(1, {{"N", count}});
auto output = output_layout.allocate();
auto binding = kernel.bind();
binding.set("X", weft::view(input, input_bytes, "dense.f32", {count}, {}, true))
       .retain("Y", output, output_layout)
       .set("scale", weft::scalar(1.25f))
       .set("bias", weft::scalar(-0.5f));
auto call = binding.prepare();
call();
```

改变 builder 不修改已经准备的 call；需要重新 prepare 才能改变地址、shape、stride 或 scalar。
`weft::sequence{&prepare_weights, &compute}` 是普通 host 调用次序。对于跨次复用的权重，应在
初始化期单独执行 `prepare_weights`，重复阶段只调用 `compute`。workspace 和 persistent state
由 host 显式保有/共享，sequence 不替调用者决定清零、复用时机或并发同步。

context 的 CPU 集合是发现时的可用范围。每次通用 call 核对当前 affinity、RVV 状态和实际使用
的扩展；sequence 在第一项写入前核对所有入口。独立存储可以并发调用同一 kernel，共享可变
存储由 host 同步。应用不得在执行期间改变 affinity 或 RVV 控制状态。

prepare/rebind 只绑定和验证存储合同，执行权限由实际执行的线程检查。
`weft_call_arguments` 可将 prepared call 的 typed ABI 参数数组借给原生 launcher；
数组、scalar 和 dimension storage 由 call 持有，直到 call_release。它必须配合同一 kernel 的
entry 使用，launcher 仍须在每个实际 worker 写入前调用能力检查。

加载 artifact 或取得 entry pointer 不执行 kernel。加载时核对该 context 中至少存在一个同时
满足所有扩展的 CPU；调用时再要求实际线程 affinity 全部属于可执行集合。因而主线程可以
加载异构核心上的扩展程序，交给具备该能力的 worker 执行，不必改变整个应用的亲和性。

需要执行线程选择时，`kernel.bind_thread()` 显式将当前线程的亲和性收窄到原集合与 artifact
全部扩展可执行集合的交集；空交集报错，不扩大调用者的 CPU 权限。该绑定在返回后保留，
调用方拥有其生命周期。`check_thread()` 仍只检查，不改变亲和性。IME adapter 在实际 worker
的节点函数选择处使用此接口，与平台首次原生 IME 指令触发的核心绑定一致。

性能敏感的框架可在逐项核对 typed ABI 后使用 `weft_kernel_get_entry`。每个有界 worker 工作段
之前调用 `weft_kernel_check_thread`，段内不能改变执行能力，kernel handle 必须覆盖全部直接调用。
这个接口不替调用者检查每次裸指针的容量；相应类型、布局与容量证明由 adapter 承担。

## llama.cpp 的调用范围

`triton.integrations.llama_cpp.install_source("/path/to/llama.cpp")` 给明确选定的上游源码树安装薄调用 hooks 和 `GGML_WEFT`
build option。工具逐项核对 GGML CPU 的调用边界；上游代码结构不匹配时报 unsupported，不能猜测
另一个替换位置。相同 hooks 可以重复安装。该操作不触碰 Weft 的只读 reference 目录。

构建时使用正常上游配置并增加：

```sh
cmake -S /path/to/llama.cpp -B /path/to/build \
  -DGGML_WEFT=ON -DWeft_DIR=/path/to/sdk/lib/cmake/Weft
cmake --build /path/to/build --target llama
```

运行前设置 `WEFT_GGML_PACKAGE=/path/to/native/package`。启用的集成必须能加载该包，且包内有
至少一个受支持的非空 binding group；缺失、ABI 不符或目标不匹配明确终止初始化。包可用
`has_binding_group(name)` 查询角色组是否存在，具名 `bindings(name)` 仍对不存在的组报错。`GGML_WEFT=OFF`
移除 adapter 依赖并保留原 GGML 调用表达式。

接入工具同时修正所支持上游的 vendor build 声明：其 IME 源码要求 C++20，且每个 IME
translation unit 必须服从上游自身的 assembler capability probes。这些纯构建修正在 ON/OFF
两种配置中相同，保留探测通过的 IME 实现。启用上游 Spacemit backend 时仍须按该目标配置
`GGML_CPU_RISCV64_SPACEMIT=ON` 与它要求的 `GGML_RV_ZBA=ON`；未提供目标指令的工具链不能假装支持。

角色名是 GGML 的 weight type 名，如 `q5_0`。adapter 核对实际链接 GGML 的 block size、record
bytes、activation Encoding、参数顺序、对齐和 `nrows`。受支持的合同是普通 RVV、单行 vec-dot，
三个 View 为 `W[K]`、`X[K]`、`Y[1]`。当前可声明 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0、Q2_K 至 Q6_K；
能声明 ABI 不表示所有配置已有性能收益，部署者只注册已选定的实现。

实际接入点位于 GGML `MUL_MAT` 和 `MUL_MAT_ID` 的 worker chunk 内部。GGML 先完成共享 activation
量化和 barrier，再调用所选 vec-dot；adapter 不重新量化，不改变输出 stride、广播、路由、chunk
划分或线程池。未注册类型继续使用 GGML 原入口，框架显式 reference 模式使用原入口。

`ggml.from_float` 可绑定 `q8_0` 或 `q8_K` activation quantizer，两个 View 为 `X[K]`、`Y[K]`，
参数顺序固定，输出 Encoding 与正常 GGML record 完全一致。调用方负责相同舍入环境；接入在
原量化调用点使用原 workspace，每个 row/chunk 只量化一次。未绑定的量化类型保留上游函数。
AMX extra-buffer 内部的 Q8_0 和 Q8_K 准备也使用同一角色与实际 worker 检查。
适配器保留该调用点的原函数；特别是未绑定的 AMX Q8_K 继续调用上游选用的
`quantize_row_q8_K_ref`，不换成另一个 CPU trait 实现。量化与后续矩阵计算之间
仍使用原 GGML barrier，计数包括在这些实际准备点执行的 Weft 调用。

更早的 CPU extra-buffer/repack/厂商 IME 路径仍由上游选择；集成不关闭其 build options。
上游设备可能公开 extra buffer type，但不支持某个 type/shape/VLEN，使用者必须遵守其
`ggml_backend_dev_supports_op` 判定。whole-matrix Weft 程序如果拥有 activation quantization
或不同的 persistent weight layout，不能注册成 vec-dot。

`ggml.ime1` 的 `q4_centered_prefill` / `q4_affine_prefill` 与
`q4_centered_decode` / `q4_affine_decode` 是独立的 packed row-tile 合同。
三个 View 为 `W[N,K]`、`Xq[M,K]`、带显式行列 stride 的 `Y[M,N]`；维度顺序为 `N,K,M`。
权重分别使用 `IMEQ4Centered_I[16]` / `IMEQ4Affine_I[16]`；prefill 的激活使用
`IMEQ8_I[4]`，decode 使用 `IMEQ8_I[1]`。adapter 在上游选择 IME1 函数时绑定显式
package entry；实际剩余行数至少四行时消费四行，否则只消费一行。这与原 IME 的
M4/M1 分派及激活布局一致，单行角色也可处理 prefill 的尾行。每次要求 32-element block、
完整 16 列 cohort，并传入原始输出行距。权重准备、激活量化、分块和线程池仍归上游。
未绑定的角色及不满足该 row-tile 合同的调用保留原 IME 函数；reference 模式
在函数选择前跳过 Weft。此绑定要求生成 artifact 实际使用 IME1，不接受仅声明扩展的 RVV 程序。

独立 `ggml-spacemit` backend 使用 `ggml.ime2/i8_prefill` 与 `ggml.ime2/i8_decode`。
权重为 `IME2Weight8Rows[32]`，激活分别为 `IME2Activation8Rows[4]` 与
`IME2Activation8Rows[1]`；原始 block32 record 分别占 34 与 38 字节，输出保留行列 stride。
这两个角色仅替换厂商 Q6_K/Q8_0 的 signed-i8 packed 调用点，保留上游权重准备、激活量化、
分块和线程调度。它们不绑定 Q4 的带 FP16 scale/accumulator 指令路径。
`install_spacemit_source(...)` 可给已有 CPU package adapter 的源码树安装该独立后端 hooks；
它逐项核对两个 i8 调用点及 graph/node 边界，并与 CPU backend 共用 package 和 counters。
构建保留上游 `GGML_SPACEMIT`，运行时仍须启用厂商 backend 并选择具有 IME2 的实际 workers。
`ggml_weft_ime2_calls()` 只累计实际调用的 Weft packed tiles，编译成功或角色注册不产生该计数。

`weft/ggml.h` 提供 registration、kernel name 和按 weight type 累积的 counters：矩阵节点数、
原 extra-buffer 节点数、Weft vec-dot 次数/结果数、已接入调用点的 activation quantize 次数/元素数。
并发线程使用独立 counters；vec-dot 在 worker chunk 入口获取函数和计数指针，原调用点
直接调用已选函数，node 结束时汇总；逐调用不查询 TLS、注册表或原子计数。
packed tile 在节点入口绑定函数与执行核心，上游 callback 捕获该 worker
binding 指针，逐 tile 不再查询 TLS；段末汇总计数。额外提供
Weft quantizer 次数及 `ggml_weft_ime1_calls()` packed tile 次数。
线程退出时归并，图运行前后差值给出该段工作的命中信息。计数只
表示实际覆盖范围，未接入的 extra-buffer 准备不记作量化命中，也不能从命中数推导模型加速。

`ggml.amx/q4_0_decode` 接入 x86 AMX buffer backend 的单行 VNNI 调用点。
三个 View 为 `W[N,K]`、`Xq[K]`、`Y[N]`，维度顺序为 `N,K`；权重
使用 `AMXQ4_0Rows[16]`，激活保留 GGML `Q8_0`。它消费上游已经构造的
16 列 cohort 和已经量化的激活，沿用原 worker 的列区间与输出地址。
同一 worker、同一 batch 的连续列区间通过一次原生调用交给 kernel，由它执行自己的列分块。
可选第四个 View `Sums[K]` 使用 `BlockSum32`：每 32 个激活元素对应一个 I16 和，
访问为 readwrite。adapter 提供每线程复用的私有存储，生成的程序在每次调用内初始化它；
没有把激活计算移出算子或模型计时。带此 workspace 的参数顺序为 `W,Xq,Y,Sums,N,K`。
包中的数学实现由共同前端和 x86 passes 编译；adapter 只核对 ABI 并直接调用。
该角色只接受 x86 SIMD 产物，不修改调用线程的 AMX tile 配置或释放 tile state。
没有注册的 weight type 继续使用正常 backend。每个 graph 的实际 worker 检查执行能力，
`ggml_weft_amx_calls()` 记录 packed 调用；计数可在测量窗口前后取差。
`ggml.amx/q4_0_prefill` 在同一 buffer backend 的多行 tile 调用点消费
`W[N,K]`、`Xq[M,K]` 和带显式 stride 的 `Y[M,N]`，维度顺序为 `N,K,M`。
调用仍由原 GGML worker 划分，传入实际剩余行数和输出行距；它与单行角色
分别选择、检查和加载，复用相同的权重布局及 SIMD 编译路线。
prefill 也可使用声明了 AMX INT8 的产物，由生成的 kernel 管理自己的 tile 配置。
动态矩阵产物一次接收同一 worker、同一输入行块的连续输出列区间，
由源码里的列遍历处理它；worker 区间、batch 边界和输入行块均保持原划分。
选中的遍历全部直接调用 Weft，因此不在每个 tile 前后重复配置原生 GGML 的 tile 状态。
完整 `M=N=32` 特化的产物只保留 `K` 维度和输出 strides，框架尾块继续由上游处理。
角色分派位于 worker 遍历入口，未选中 package 的原生热循环不包含间接调用分支。
使用 AMX 的产物完成该 worker 遍历后，adapter 使同一线程的 GGML tile 配置缓存失效。
后续未替换的 AMX 节点会重新加载原生配置；不在每个替换 tile 前后重复保存和恢复。
纯 SIMD 产物不改变该缓存。混合替换因此保留正常 GGML 与产物各自的 tile 状态合同。

动态 prefill 还可显式接收第四个 View `Scales[M,K]`，使用
`BlockScale32Rows[32]`：每 32 个逻辑元素对应一个 F32 字段，跨 32 行按
4 字节字段交错。调用顺序为 `W,Xq,Y,Scales,N,K,M,stride_m,stride_n`。
adapter 为每个 worker 保留独立的对齐存储；源程序在每次调用中把 activation
scale 精确扩宽为 F32，再跨输出列复用。填充仍计入 kernel 和模型时间，
没有跨调用缓存 activation 内容。该 ABI 要求完整 32 行 cohort，框架只在
输入行数为 32 的倍数时选择此绑定，每次交付一个 32 行块；尾行继续使用
正常 GGML 路径。普通三参数动态 prefill 的尾行合同不受此 workspace 限制。

`ggml.amx/q4_K_decode` 使用相同的单行 `W,Xq,Y,N,K` ABI，消费
`AMXQ4_KRows[16]` 与 `Q8_K`。每 256 元素的 packed 权重记录为 148 字节：
低位数值按 4 字节 chunk 交错，8 个 scale 按 1 字节交错，8 个 min 按
2 字节交错，两个 FP16 block scale 各按 2 字节交错。这是 GGML 正常
AMX buffer 已生成的布局，区别于其 144 字节 Q4_K 原始记录。权重准备
仍归 GGML，kernel 在调用内读取各量化组并计算整数部分与缩放。
该角色没有额外 workspace；按 weight type 和阶段分别保存绑定及 worker
检查状态，Q4_0 与 Q4_K 可同时注册而不共享错误的执行许可。

`ggml.amx/q4_K_prefill` 在多行调用点消费同一 `AMXQ4_KRows[16]` 权重与
`Q8_K[M,K]` 激活，输出为带显式 stride 的 `Y[M,N]`。支持动态 `N,K,M`
矩阵 ABI 或固定 32×32、仅有运行时 `K` 的 tile ABI。固定产物只接完整
32 行块；动态产物沿用已有 worker 列区间。它没有 Q4_0 专用的 scale
workspace，权重准备、激活量化和线程分工继续由正常 GGML 完成。

`ggml.amx/q6_K_decode` 使用同一单行 ABI 和 `Q8_K` 激活，权重为
`AMXQ6_KRows[16]`。每 256 元素记录含 128 字节低位、64 字节高位、
16 个 I8 scale 和一个 FP16 block scale，共 210 字节；低位与高位各按
4 字节 chunk 交错，scale 按 1 字节、block scale 按 2 字节交错。
绑定独立检查该布局与实际 worker 能力，直接消费 GGML 的既有 packed buffer。

`ggml.amx/q6_K_prefill` 使用上述 Q6_K 权重和 Q8_K 多行激活，沿用 Q4_K
prefill 的动态矩阵或固定 32×32 typed ABI。两个 prefill 角色可以与各自
decode、Q8_K 量化角色同时绑定；未绑定类型及不完整的固定行块保持上游路径。

worker 在每次 graph 的 NUMA/核心设置后开启新的能力检查作用域。无外部 abort/custom
callback 的原生 graph 保持 RVV permission，原生 IME 的亲和性收窄也保持已有检查成立；
每个实际使用的 vec-dot/quantizer 入口在此作用域内只检查一次，packed IME 入口也只做
一次执行能力检查与必要的亲和性收窄。缓存按具体入口区分，每次 graph 开始时清空。图含外部 callback 时，
每个矩阵节点仍检查；graph 结束后的核心恢复不能把检查状态带入下一次 graph。

## 测量边界

原生单实例、完整 grid/OpenMP launch 和模型 E2E 分开计时。
通过独立 Weft Bench 验证导出产物、加载、数值与角色命中；prefill/decode 分开记录。
正常 llama.cpp 的 repack、IME、OpenMP 等能力继续保留，固定 baseline 仅在测量合同变化时补测。
集成相关源码位于 integrations/llama.cpp，Python 安装/角色接口位于
python/triton/integrations/llama_cpp.py，具名产物集合位于 python/triton/runtime/package.py。
