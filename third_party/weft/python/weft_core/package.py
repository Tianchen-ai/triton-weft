"""Frontend-neutral native artifacts and application-owned role bindings."""
import json
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping
from .artifact import NativeArtifact


def sdk_prefix():
    from weft_core import sdk_path
    prefix = sdk_path()
    for name in ("include/weft/runtime.h", "include/weft/runtime.hpp",
                 "lib/libweft_runtime.so", "lib/cmake/Weft/WeftConfig.cmake"):
        if not (prefix / name).is_file():
            raise FileNotFoundError(f"the installed Weft native SDK is incomplete: {prefix / name}")
    return prefix


class Package:
    """A relocatable collection of already compiled, explicitly selected kernels."""

    load_artifact = staticmethod(NativeArtifact.load)

    def __init__(self, directory):
        load = self.load_artifact
        self.path = Path(directory).expanduser().absolute()
        description = json.loads((self.path / "package.json").read_text())
        if description["kind"] != "weft-kernel-package":
            raise ValueError("expected a Weft native kernel package")
        self.kernels = {}
        for entry in description["kernels"]:
            name, relative = entry["name"], Path(entry["directory"])
            if name in self.kernels or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("duplicate package name or non-relative artifact path")
            self.kernels[name] = load(self.path / relative)
        if not self.kernels:
            raise ValueError("package contains no native kernels")
        self.bindings = description["bindings"]
        self._validate(self.kernels, self.bindings)

    @staticmethod
    def _validate(kernels, bindings):
        if not kernels or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name) for name in kernels):
            raise ValueError("package requires named kernels using letters, digits, underscore or hyphen")
        for kernel in kernels.values():
            if not callable(getattr(kernel, "export", None)) or kernel.metadata.target.backend != "weft":
                raise TypeError("native package entries must be compiled Weft CPU artifacts")
        if len({(k.metadata.target.arch, k.metadata.target.abi, k.metadata.target.vlen_bits) for k in kernels.values()}) != 1:
            raise ValueError("one native package must use a common execution ABI and VLEN")
        for group, mapping in bindings.items():
            if not isinstance(group, str) or not group or not isinstance(mapping, Mapping):
                raise ValueError("bindings require named role-to-kernel mappings")
            for role, name in mapping.items():
                if not isinstance(role, str) or not role or name not in kernels:
                    raise ValueError("binding role must name an existing package entry")

    @classmethod
    def create(cls, directory, kernels, *, bindings=None):
        registrations = {} if bindings is None else {group: dict(mapping) for group, mapping in bindings.items()}
        cls._validate(kernels, registrations)
        destination = Path(directory).expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix=".weft-package-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "package"
            staging.mkdir()
            entries = []
            for name, kernel in kernels.items():
                relative = "kernels/" + name
                kernel.export(staging / relative)
                entries.append({"name": name, "directory": relative})
            (staging / "package.json").write_text(json.dumps({"kind": "weft-kernel-package",
                "producer": getattr(next(iter(kernels.values())).metadata, "producer", "triton"), "kernels": entries, "bindings": registrations}, indent=2))
            os.rename(staging, destination)
        return cls(destination)
