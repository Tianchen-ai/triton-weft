"""Triton object loading over the shared native package implementation."""
from weft_core.package import Package as NativePackage, sdk_prefix
from .weft.artifact import load


class Package(NativePackage):
    load_artifact = staticmethod(load)
