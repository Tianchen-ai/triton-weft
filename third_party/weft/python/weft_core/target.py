"""CPU descriptors and capability queries, independent of a frontend driver."""
from dataclasses import asdict, dataclass

from . import compiler
from .options import WeftOptions


@dataclass(frozen=True)
class CPUTarget:
    backend: str
    arch: str
    march: str
    abi: str
    vlen_bits: int
    matrix_extension: str = "none"


def extension_catalog():
    return compiler.extension_catalog()


def target_capabilities(target: CPUTarget, *, matrix_extension=None):
    if not isinstance(target, CPUTarget) or target.backend != "weft" or target.arch not in ("riscv64", "x86_64"):
        raise TypeError("target_capabilities requires a Weft RISC-V or x86-64 CPUTarget")
    matrix_extension = target.matrix_extension if matrix_extension is None else matrix_extension
    if target.arch == "x86_64":
        if matrix_extension != "none":
            raise ValueError("x86 matrix capabilities belong to the target ISA feature set")
        return compiler.x86_target_capabilities(target.march, target.abi, target.vlen_bits)
    if not isinstance(matrix_extension, str):
        raise TypeError("matrix_extension must be a registered extension name or 'none'")
    options = asdict(WeftOptions(matrix_extension=matrix_extension))
    options.update(march=target.march, abi=target.abi, vlen_bits=target.vlen_bits)
    return compiler.target_capabilities(options)


def native_capabilities():
    return compiler.native_target()

