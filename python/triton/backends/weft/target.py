"""Inspect CPU compiler targets and native execution capabilities.

Catalog and explicit-target queries work on a cross-build host and require no
C compiler, sysroot or device. Native discovery reports the current CPU set;
it neither changes affinity nor grants permission to execute an extension.
"""
from dataclasses import asdict

from triton._C.libtriton import weft
from triton.backends.compiler import CPUTarget
from .compiler import WeftOptions

__all__ = ["extension_catalog", "target_capabilities", "native_capabilities"]


def extension_catalog() -> dict:
    """Return the RISC-V local extensions built into this compiler and SDK."""
    return weft.extension_catalog()


def target_capabilities(target: CPUTarget, *, matrix_extension: str | None = None) -> dict:
    """Describe an explicit build target using the compiler's target validator.

    ``matrix_extension`` has the same meaning as the backend option. A positive
    result proves that this compiler implements the target contract, not that
    the host CPU or selected C toolchain can execute or assemble it.
    """
    if not isinstance(target, CPUTarget) or target.backend != "weft" or target.arch not in ("riscv64", "x86_64"):
        raise TypeError("target_capabilities requires a Weft RISC-V or x86-64 CPUTarget")
    if matrix_extension is None:
        matrix_extension = target.matrix_extension
    if target.arch == "x86_64":
        if matrix_extension != "none":
            raise ValueError("x86 matrix capabilities belong to the target ISA feature set")
        return weft.x86_target_capabilities(target.march, target.abi, target.vlen_bits)
    if not isinstance(matrix_extension, str):
        raise TypeError("matrix_extension must be a registered extension name or 'none'")
    options = asdict(WeftOptions(matrix_extension=matrix_extension))
    options.update(march=target.march, abi=target.abi, vlen_bits=target.vlen_bits)
    return weft.target_capabilities(options)


def native_capabilities() -> dict:
    """Discover ISA facts and eligible extension CPUs in the current affinity.

    Actual workers still undergo the SDK's execution checks at launch time.
    RISC-V and x86-64 Linux report their own native target; a cross-build host
    does not emulate execution capabilities for another architecture.
    """
    return weft.native_target()
