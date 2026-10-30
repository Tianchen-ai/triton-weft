# Third-party notices

Weft's own code is distributed under Apache-2.0; this does not replace the
licenses of dependencies or retained source material.

## LLVM and MLIR

Weft builds against LLVM/MLIR. The compiler links MLIR components and the native
runtime uses LLVM support/target facilities. Weft shares the revision selected
by Triton's [SDK manifest](../../cmake/llvm-info.json). The upstream license is
Apache-2.0 with LLVM exceptions, with legacy notices retained in
[`licenses/LLVM.txt`](licenses/LLVM.txt). Bundled Clang also retains the license
supplied with the actual toolchain.

Notices are installed under the native SDK's `licenses/` directory in the
Triton wheel. The declared Clang and OpenMP runtime can be bundled; libc and
other platform dependencies come from the declared target environment.
Distributors must preserve the notices required by their chosen components.

## Adapted GGML thread execution

The shared native runtime includes a compact adaptation of GGML's worker
wakeup, polling and barrier implementation from llama.cpp revision
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. It executes checked native ranges
without GGML tensors, graphs or library linkage. Copyright and MIT terms are
retained in `lib/Runtime/ggml-threadpool-LICENSE`, installed with the SDK.

## Optional GGML / llama.cpp integration

The optional adapter and comparison examples build against the user's explicit
[llama.cpp checkout](https://github.com/ggml-org/llama.cpp), which retains its
[MIT license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) and upstream
file notices. Apart from the adapted threading code above, the GGML library
is not bundled and is not a compiler or runtime fallback.

## Optional dependencies and references

NumPy and PyTorch are optional, independently installed dependencies; their
distributions retain their own licenses. Triton is the enclosing MIT-licensed
source distribution; see the root [NOTICE](../../NOTICE). TileLang and hardware
references inform design comparisons but are not linked by Weft. No local
reference tree is required to build or use the installed package.
