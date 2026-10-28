# 导出 kernel，并从 C++ 调用

原生 SDK 加载 Triton 导出的动态库并绑定 typed ABI。
C++ 应用持有数据，调用编译好的函数。

## 1. 导出单实例 View kernel

View 教程具有动态 extent `N`，以及 scalar 参数 `scale`、`bias`。
这里编译一个覆盖 64 元素的程序：

```sh
python -m examples.weft.language.views --size 64 --threads 1 --output /tmp/affine-native
python -c "from triton.runtime.package import sdk_prefix; print(sdk_prefix())"
```

第一条命令生成 `kernel.so`、`kernel.h`、`manifest.json`、
`triton.json` 和诊断材料；第二条命令输出 SDK 目录。

## 2. 在 C++ 中加载并绑定

完整程序位于
[`examples/weft/deployment/native_call.cpp`](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/deployment/native_call.cpp)，
核心调用如下：

```cpp
weft::context device;
weft::kernel kernel(device, artifact_directory);
auto binding = kernel.bind();
binding.set("X", weft::view(x.data(), x.size() * sizeof(float),
                           "dense.f32", {x.size()}, {}, true))
       .set("Y", weft::view(y.data(), y.size() * sizeof(float),
                           "dense.f32", {y.size()}))
       .set("scale", weft::scalar(1.25f))
       .set("bias", weft::scalar(-0.5f));
auto call = binding.prepare();
call();
```

Prepare 检查容量、维度、Encoding、access 和 alias 合同；
call 持有准备好的 metadata，数组由应用保持有效。
导出顶层 manifest 描述单实例入口，因此本例向 64 元素程序传入 64 个元素。

## 3. 构建并运行

```sh
cmake -S examples/weft/deployment -B /tmp/weft-native-build \
  -DWeft_DIR=/printed/sdk/path/lib/cmake/Weft
cmake --build /tmp/weft-native-build
/tmp/weft-native-build/weft-native-call /tmp/affine-native
```

前四个输出是 `-0.5 0.75 2 3.25`。
应用链接 `Weft::Runtime`；runtime 加载已导出产物，
SDK 和 artifact 应匹配实际执行目标。

## 具名集合与 worker 接入

`triton.Package.create` 将已选 kernel 组成应用具名集合。
`weft::package` 加载集合，`get_kernel(name)` 获取入口。
`kernel.storage_layout(...)` 描述分配要求，
`weft::sequence` 顺序调用显式排列的 prepared call。

已有 worker pool 可调用单实例入口，并在有界工作段前检查 worker 能力。
独立 grid metadata 描述 OpenMP launcher；
[llama.cpp](../integrations/llama-cpp.md)使用前一种入口。

[完整 SDK 所有权、绑定与 worker 合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/native-sdk.md)。
