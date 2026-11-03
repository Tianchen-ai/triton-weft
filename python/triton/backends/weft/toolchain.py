"""Triton cache adapter for the shared native toolchain."""
from pathlib import Path
from weft_core.toolchain import Toolchain, native_gcc_runtime, gcc_runtime_identity
from .diagnostics import compiler_call


def compile_native(toolchain, source, metadata, target):
    from triton.runtime.cache import get_cache_manager
    cache = get_cache_manager(metadata["hash"])
    result = compiler_call(toolchain.compile, source, metadata, target, cache.cache_dir)
    metadata["runtime_files"] = {name: cache.put(Path(path).read_bytes(), name)
                                 for name, path in metadata["runtime_files"].items()}
    return result
