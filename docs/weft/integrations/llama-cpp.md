# From a Python kernel to a llama.cpp call

Python supplies compiler input. GGML workers execute an exported native library.
There is no Python interpreter, compiler subprocess or additional OpenMP team
inside the replaced call.

The complete tutorial is `examples/weft/llama_cpp.py`. It declares Q8_0 storage,
reduces each 32-element signed byte block into I32, applies both FP16 scales in
F32, and accumulates a row. Its public arguments are `W[K], X[K], Y[1]` with one
dynamic K, matching the `ggml.vec_dot/q8_0` role. It is a readable deployment
example, not a promise of acceleration over an optimized native dot.

## Compile and bind

```sh
python examples/weft/llama_cpp.py /path/to/new-package --profile native
```

For a cross build, pass `--profile k1` (or the selected target) and `--toolchain`
with the declared compiler/sysroot JSON. The script calls `triton.compile`, then
`triton.integrations.llama_cpp.create_package`. The integration checks Encoding,
parameter order, strides, numerical extent and target requirements before export.
An arbitrary matmul cannot be bound to this row-dot role.

The package contains the `.so`, typed header, ABI/target metadata and required
native libraries. Generated C and IR are inspection material. Deployment loads
the existing `.so` and does not rebuild that C.

## Build the adapter into the application

Choose the supported llama.cpp source version documented by Bench's reference
catalog. Keep its normal CPU, repack, vendor and OpenMP options.

```python
from triton.integrations.llama_cpp import install_source
from triton.runtime.package import sdk_prefix
install_source("/path/to/llama.cpp")
print(sdk_prefix())
```

```sh
cmake -S /path/to/llama.cpp -B /path/to/llama-build \
  -DGGML_WEFT=ON -DWeft_DIR=/printed/sdk/path/lib/cmake/Weft
cmake --build /path/to/llama-build --target llama-cli llama-bench
WEFT_GGML_PACKAGE=/path/to/new-package \
  /path/to/llama-build/bin/llama-cli -m /path/to/q8-model.gguf -p 'Hello'
```

The adapter validates the actual GGML record layout, loads the package during
initialization, and binds the named row-dot call. GGML still prepares weights,
quantizes activations and assigns work to its existing workers. A backend that
uses a different packed matrix call needs its corresponding IME/AMX role instead.

## Check actual replacement

Use the independent Bench `prepare-model` and `e2e` commands to reuse a configured
model protocol, logit oracle, role mapping and normal baseline. Its observer reads
the adapter counters before and after each phase. A loaded package with no calls
does not demonstrate a model replacement or acceleration. Prefill and decode are
reported separately. The detailed [native SDK contract](../compiler/native-sdk.md)
lists supported hooks, layouts, counter APIs and thread boundaries.
