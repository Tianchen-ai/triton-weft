# Export a kernel and call it from C++

The native SDK loads the library exported by Triton and binds its typed ABI.
A C++ application owns the data and invokes the compiled function.

## 1. Export a single-instance View kernel

The View tutorial has runtime extent `N` and scalar parameters `scale` and `bias`.
For this C++ example, compile a single program covering 64 elements:

```sh
python -m examples.weft.language.views --size 64 --threads 1 --output /tmp/affine-native
python -c "from triton.runtime.package import sdk_prefix; print(sdk_prefix())"
```

The first command writes `kernel.so`, `kernel.h`, `manifest.json`,
`triton.json` and inspection material. The second prints the SDK directory.

## 2. Load and bind in C++

The complete program is
[`examples/weft/deployment/native_call.cpp`](https://github.com/Tianchen-ai/triton-weft/blob/main/examples/weft/deployment/native_call.cpp).
Its central sequence is:

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

Preparation validates capacity, dimensions, Encoding, access and alias contracts.
The call owns its prepared metadata; the application's arrays remain alive.
The exported top-level manifest describes the single-instance entry, so this
example passes 64 elements to the 64-element program.

## 3. Build and run

```sh
cmake -S examples/weft/deployment -B /tmp/weft-native-build \
  -DWeft_DIR=/printed/sdk/path/lib/cmake/Weft
cmake --build /tmp/weft-native-build
/tmp/weft-native-build/weft-native-call /tmp/affine-native
```

The first four values are `-0.5 0.75 2 3.25`.
The application links `Weft::Runtime`. Runtime loading uses the exported artifact;
the SDK and the artifact must match the executing target.

## Collections and worker integration

`triton.Package.create` groups selected kernels under application names.
`weft::package` loads that collection; `get_kernel(name)` retrieves an entry.
`kernel.storage_layout(...)` describes allocation requirements, while
`weft::sequence` invokes explicitly ordered prepared calls.

An existing worker pool can use single-instance entries and validate worker
capabilities before a bounded call segment. The separate grid metadata describes
the OpenMP launcher. [llama.cpp](../integrations/llama-cpp.md) uses the former.

[Full SDK ownership, binding and worker contract](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/native-sdk.md).
