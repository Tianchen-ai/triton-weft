# Targets, execution cores and engines

Choose three related things explicitly: **where the process runs**, **which
instructions compilation may use**, and **which route a contraction requests**.

## Discover and select cores

```sh
python -m triton.tools.cpu --list
python -m triton.tools.cpu --profile k3-a100 --cpus 8-15 application.py
python -m triton.tools.cpu --profile k3-x100 --cpus 0-7 application.py
```

The K3 A100 launcher registers its PID through `/proc/set_ai_thread`, sets CPU
affinity, then starts the application. Target discovery runs on the selected core
group. K3 X100 has VLEN 256; A100 has VLEN 1024. Select a homogeneous group.

## Construct a compilation target

```python
from triton import cpu

print(cpu.devices())
print(cpu.target())          # native capability discovery
print(cpu.target("k3-a100")) # explicit compilation profile
```

| Profile | Architecture / vector capability | Matrix capability |
|---|---|---|
| `sg2044` | RVV, VLEN 128 | — |
| `k1` | RVV, VLEN 256 | IME1 |
| `k3-x100` | RVV, VLEN 256 | — |
| `k3-a100` | RVV, VLEN 1024 | IME2 |
| `x86-avx2` | x86-64-v3, 256-bit SIMD | — |
| `x86-avx512` | x86-64-v4, 512-bit SIMD | — |
| `x86-amx` | x86-64-v4, 512-bit SIMD | AMX INT8/BF16 |

Pass this target to `triton.compile(ASTSource(...), target=...)` for AOT.
The profile expands to capability facts; `CPUTarget` also supports explicit ISA,
ABI, vector-length and extension fields.

## Constrain a contraction

```python
tl.dot(a, b, acc)                    # legal matrix implementation first, then vector
tl.dot(a, b, acc, engine="matrix")   # require a legal matrix implementation
tl.dot(a, b, acc, engine="vector")   # choose from vector implementations
```

The default keeps the source dtype and accumulator contract. Vector implementations
include RVV, SIMD and VNNI. A kernel may combine vector decoding, matrix arithmetic
and vector postprocessing. Target passes choose layouts and resources within each
operation's route.

Inspect `compiled.metadata.kernel_abi["used_extensions"]` and physical IR.
Artifact requirements identify the executable capabilities; a conditional matrix
path's measured invocation is recorded separately by Bench observers.

Grid size and `num_cpu_threads` remain independent. Runtime checks the actual
workers; AMX permission is handled for artifacts requiring AMX.
