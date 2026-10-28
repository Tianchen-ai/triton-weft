# 部署到 llama.cpp

完整部署过程为：

**Triton Python 源码 → 原生库与 typed ABI → GGML 角色绑定 → worker 调用。**

本教程导出一个 Q8_0 row dot。GGML 准备模型数据、划分 worker chunk，
绑定的函数负责匹配的局部 dot 计算。

## 1. 阅读角色对应的源程序

[`deployment/llama_cpp.py`](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/deployment/llama_cpp.py)
从 `triton.integrations.ggml_formats` 导入 `Q8_0`。
每个 32 元素 block 对 signed-byte 字段求 I32 dot，
将两个 FP16 scale 在 F32 中相乘，并更新整行 total。

参数为 `W[K], X[K], Y[1]`，匹配 `ggml.vec_dot/q8_0`，
两个输入都具有实际 GGML Encoding。

## 2. 编译并绑定

```sh
python -m examples.weft.deployment.llama_cpp /tmp/q8-package --profile native
```

交叉编译可选择 `--profile k1 --toolchain toolchain.json`。
脚本先调用 `triton.compile`，随后：

```python
create_package(output, {"q8_dot": compiled},
               bindings={"ggml.vec_dot": {"q8_0": "q8_dot"}})
```

Integration 检查布局、shape、参数顺序和目标要求。
Package 保存编译好的库及调用点所需 metadata。

## 3. 构建应用 adapter

选择 Bench [参考目录](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/src/weft_bench/catalogs/references.json)
登记的 llama.cpp revision，在选定 checkout 中安装接入：

```python
from triton.integrations.llama_cpp import install_source
from triton.runtime.package import sdk_prefix
install_source("/path/to/llama.cpp")
print(sdk_prefix())
```

保持应用正常后端配置，并加入 adapter：

```sh
cmake -S /path/to/llama.cpp -B /path/to/llama-build \
  -DGGML_WEFT=ON -DWeft_DIR=/printed/sdk/path/lib/cmake/Weft
cmake --build /path/to/llama-build --target llama-cli llama-bench
WEFT_GGML_PACKAGE=/tmp/q8-package \
  /path/to/llama-build/bin/llama-cli -m /path/to/q8-model.gguf -p 'Hello'
```

GGML 在初始化期加载和绑定 package，
worker 沿用已有输入布局、输出合同和线程池，直接调用原生函数。

## 4. 查看实际调用与性能

Adapter 提供登记角色的调用计数；Bench 记录每个测量窗口内的差值，
并单独执行登记的 logits 检查。
使用 [prepare-model 与 e2e](../experiments/reproduce.md)运行所选 package，
匹配正常参考，并分别测量 prefill／decode。

优化 corpus 还提供多行、IME 与 AMX 程序及对应 packed role 合同。
通过 `weft-bench list --kind replacements` 和模型 package 配方选择角色。

[完整角色、布局与 worker 合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/native-sdk.md)。
