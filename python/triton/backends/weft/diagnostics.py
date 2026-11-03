"""Translate shared compiler diagnostics at the Triton API boundary."""
from weft_core.compiler import CompilationError


def compiler_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except CompilationError as error:
        if error.diagnostic.get("kind") != "weft-compilation-failure":
            raise
        from triton.compiler.errors import BackendCompilationError
        raise BackendCompilationError(error.diagnostic) from error
