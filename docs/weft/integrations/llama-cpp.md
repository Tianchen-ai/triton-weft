# Deploy into llama.cpp

The deployment path is:

**Triton Python source → native library and typed ABI → GGML role binding → worker call.**

The tutorial exports a Q8_0 row dot. GGML prepares the model data and assigns worker
chunks; the bound function computes the matching local dot.

## 1. Read the role's source program

[`deployment/llama_cpp.py`](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/deployment/llama_cpp.py)
imports `Q8_0` from `triton.integrations.ggml_formats`.
For each 32-element block it computes an I32 dot of the signed-byte fields,
multiplies the two FP16 scales in F32 and updates the row total.

Its signature is `W[K], X[K], Y[1]`, matching `ggml.vec_dot/q8_0`.
Both arguments carry the actual GGML Encoding.

## 2. Compile and bind

```sh
python -m examples.weft.deployment.llama_cpp /tmp/q8-package --profile native
```

For cross compilation select, for example, `--profile k1 --toolchain toolchain.json`.
The script calls `triton.compile`, then:

```python
create_package(output, {"q8_dot": compiled},
               bindings={"ggml.vec_dot": {"q8_0": "q8_dot"}})
```

The integration validates layout, shape, argument order and target requirements.
The package contains the compiled library and the metadata needed at the call site.

## 3. Build the application adapter

Choose the llama.cpp revision registered by Bench's
[reference catalog](https://github.com/Tianchen-ai/triton-weft-bench/blob/main/src/weft_bench/catalogs/references.json).
Install the integration into that selected checkout:

```python
from triton.integrations.llama_cpp import install_source
from triton.runtime.package import sdk_prefix
install_source("/path/to/llama.cpp")
print(sdk_prefix())
```

Keep the application's normal backend settings and add the adapter:

```sh
cmake -S /path/to/llama.cpp -B /path/to/llama-build \
  -DGGML_WEFT=ON -DWeft_DIR=/printed/sdk/path/lib/cmake/Weft
cmake --build /path/to/llama-build --target llama-cli llama-bench
WEFT_GGML_PACKAGE=/tmp/q8-package \
  /path/to/llama-build/bin/llama-cli -m /path/to/q8-model.gguf -p 'Hello'
```

GGML loads and binds the package during initialization. Its workers call the
native function using the existing input layout, output contract and thread pool.

## 4. Inspect actual calls and performance

The adapter exposes counters for the registered roles. Bench records differences
within each measurement window and checks registered logits separately.
Use [prepare-model and e2e](../experiments/reproduce.md) to run the selected package
with matching normal references and separate prefill/decode measurements.

The optimized corpus also provides multi-row, IME and AMX programs with their
specific packed role contracts. Choose roles from `weft-bench list --kind replacements`
and the model's package recipe.

[Full role, layout and worker contracts](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/native-sdk.md).
