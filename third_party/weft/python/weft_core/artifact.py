"""Frontend-neutral native artifacts; loading never invokes the compiler."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace

from .native import NativeModule, NativeLauncher, manifest
from .launcher import GRID_DECLARATION
from .target import CPUTarget


def header(metadata):
    declarations = metadata.kernel_abi["declaration"] + "\n" + metadata.single_instance_abi["declaration"] + "\n"
    for argument in metadata.kernel_abi["arguments"]:
        pack = argument.get("artifact_pack")
        if pack:
            declarations += f"size_t {pack['size_symbol']}(size_t rows, size_t row_elements);\n"
            declarations += f"void {pack['pack_symbol']}(const void *source, void *destination, size_t rows, size_t row_elements);\n"
    return ('#pragma once\n#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n'
            '#ifdef __cplusplus\n#define WEFT_RESTRICT __restrict__\nextern "C" {\n'
            '#else\n#define WEFT_RESTRICT restrict\n#endif\n'
            + re.sub(r"\brestrict\b", "WEFT_RESTRICT", declarations)
            + GRID_DECLARATION + '\n#ifdef __cplusplus\n}\n#endif\n#undef WEFT_RESTRICT\n')


class NativeArtifact:
    """Owns an already compiled library and its typed argument/target contract."""

    def __init__(self, directory, metadata, *, library="kernel.so", owner=None):
        self.directory = Path(directory).absolute()
        self.library = self.directory / library
        if not self.library.is_file():
            raise FileNotFoundError(self.library)
        self.metadata = metadata
        self._owner = owner
        self._module = None
        self._launcher = None

    @classmethod
    def create(cls, binary, metadata, diagnostics):
        owner = tempfile.TemporaryDirectory(prefix="weft-native-")
        root = Path(owner.name)
        (root / "kernel.so").write_bytes(binary)
        for extension, source in diagnostics.items():
            (root / ("kernel." + extension)).write_text(source)
        result = cls(root, metadata, owner=owner)
        result.write_metadata(root)
        return result

    @classmethod
    def load(cls, directory, *, library="kernel.so"):
        root = Path(directory).absolute()
        values = json.loads((root / "kernel.native.json").read_text())
        values["target"] = CPUTarget(**values["target"])
        for name, relative in values["runtime_files"].items():
            dependency = (root / relative).resolve()
            if not dependency.is_relative_to(root.resolve()) or not dependency.is_file():
                raise ValueError(f"native runtime dependency must exist inside the artifact: {relative}")
            values["runtime_files"][name] = str(dependency)
        return cls(root, SimpleNamespace(**values), library=library)

    def write_metadata(self, directory):
        root = Path(directory)
        values = dict(vars(self.metadata))
        values["target"] = asdict(self.metadata.target)
        values["runtime_files"] = {}
        for name, source in self.metadata.runtime_files.items():
            path = root / name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("native runtime dependencies require relative artifact paths")
            path.parent.mkdir(parents=True, exist_ok=True)
            if Path(source).resolve() != path.resolve():
                shutil.copyfile(source, path)
            values["runtime_files"][name] = name
        (root / "kernel.native.json").write_text(json.dumps(values, indent=2))
        description = manifest(self.metadata)
        description["kernel"] = self.metadata.single_instance_abi
        description["dispatch"] = "triton_weft_instance_dispatch"
        description["grid"] = {**manifest(self.metadata), "launcher": "triton_weft_grid"}
        (root / "manifest.json").write_text(json.dumps(description, indent=2))
        (root / "kernel.h").write_text(header(self.metadata))

    def export(self, directory):
        destination = Path(directory).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix=".weft-artifact-", dir=destination.parent) as temp:
            staging = Path(temp) / "artifact"
            staging.mkdir()
            for source in self.directory.glob("kernel.*"):
                if source.is_file():
                    shutil.copyfile(source, staging / source.name)
            if self.library.name != "kernel.so":
                shutil.copyfile(self.library, staging / "kernel.so")
            self.write_metadata(staging)
            os.rename(staging, destination)
        return destination

    def launch(self, grid, arguments):
        if self._launcher is None:
            self._module = NativeModule(self.library, self.metadata)
            self._launcher = NativeLauncher(self._module, self.metadata)
        if len(grid) > 3:
            raise ValueError("CPU grid has at most three dimensions")
        grid = (*grid, *(1 for _ in range(3 - len(grid))))
        self._launcher(*grid, None, self._module.entry.value, self.metadata, None, None, None, *arguments)
