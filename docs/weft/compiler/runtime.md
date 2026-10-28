# JIT, cache and autotune

A Weft kernel uses Triton's `JITFunction`, specialization, cache,
`CompiledKernel` and autotuner. The CPU backend supplies target discovery,
native compilation and a grid launcher.

## The first and subsequent calls

```python
compiled = kernel[(instances,)](X, Y, B=64, num_cpu_threads=2)
kernel[(instances,)](X, Y, B=64, num_cpu_threads=2)
```

The first call resolves the target and argument contracts, specializes source,
compiles and loads the native program. The second reuses the specialization.
Dependencies include helpers, Encoding/derived declarations and source constants.
Runtime shapes, strides and scalars are passed through the typed ABI.

Set `TRITON_CACHE_DIR` for the application's compilation cache. Toolchain,
target features, ABI and compile/link options participate in compilation identity.
An explicit sysroot uses the application's declared `sysroot_identity`.

## Grid and threads

Grid specifies independent instances. `num_cpu_threads` specifies OpenMP execution.
A single instance calls native code directly; multiple instances with one thread
use a native loop; multiple threads use the native OpenMP launcher.
`num_cpu_threads=0` uses OpenMP's current setting.

The application partitions data with `tl.program_id`. Each instance runs an
ordered source program. Frameworks with existing workers can use the exported
single-instance entry, as described in the [native SDK](native-sdk.md).

## Select source and physical configurations

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

`kwargs` binds source parameters; `backend_options` binds physical parameters
such as `pipeline_depth` (1–4); `num_cpu_threads` configures the launcher.
Triton supplies candidate enumeration, pruning, hooks, benchmarking and cached choice.
Each candidate passes numerical, effect and resource legality checks.

Autotune trials use private storage while retaining overlapping aliases.
Mutable inputs are restored between trials; the selected kernel then executes
once on the caller's storage.

## CPU buffers

NumPy buffers and CPU Torch tensors borrow their existing storage. View metadata
preserves shape, nonnegative declared strides, offset and aliases. Packed storage
uses `tl.View(bytes, encoding=..., shape=...)`.
The caller keeps storage live during execution.

## Export and reload

```python
compiled.export("/tmp/selected-kernel")
loaded = triton.load("/tmp/selected-kernel")
loaded[(instances,)](X, Y)
```

Use the runtime arguments of the exported kernel; constexpr values are already fixed.
Export to a new directory. The artifact contains the native library, header,
manifest, Triton metadata and runtime dependencies.
See the [complete runnable export example](../getting-started/examples.md).

[Detailed runtime contract](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/compiler/runtime.md).
