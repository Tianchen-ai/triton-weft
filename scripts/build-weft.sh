#!/usr/bin/env bash
set -euo pipefail

source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_command=${PYTHON:-python3}
export LLVM_SYSPATH=${LLVM_SYSPATH:?Set LLVM_SYSPATH to the pinned LLVM/MLIR SDK}
export TRITON_WEFT_CLANG_ROOT=${TRITON_WEFT_CLANG_ROOT:-$LLVM_SYSPATH}
if [[ -z ${TRITON_WEFT_OPENMP_INCLUDE_DIR:-} ]]; then
    TRITON_WEFT_OPENMP_INCLUDE_DIR=$($python_command - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['LLVM_SYSPATH'])
candidates = [root / 'include', *sorted((root / 'lib' / 'clang').glob('*/include'))]
match = next((path for path in candidates if (path / 'omp.h').is_file()), None)
if match is None:
    raise SystemExit('Set TRITON_WEFT_OPENMP_INCLUDE_DIR to the directory containing omp.h')
print(match)
PY
)
    export TRITON_WEFT_OPENMP_INCLUDE_DIR
fi
if [[ -z ${TRITON_WEFT_OPENMP_LIBRARY:-} ]]; then
    TRITON_WEFT_OPENMP_LIBRARY=$($python_command - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['LLVM_SYSPATH'])
matches = sorted((root / 'lib').rglob('libomp.so'))
if not matches:
    raise SystemExit('Set TRITON_WEFT_OPENMP_LIBRARY to the matching OpenMP runtime')
print(matches[0])
PY
)
    export TRITON_WEFT_OPENMP_LIBRARY
fi
export CC=${CC:-gcc}
export CXX=${CXX:-g++}
export TRITON_BUILD_DIR=${TRITON_BUILD_DIR:-$source_root/.build}
export TRITON_BUILD_PROTON=${TRITON_BUILD_PROTON:-OFF}
if [[ $(uname -m) == riscv64 && -z ${LDFLAGS:-} ]]; then
    export LDFLAGS=-Wl,--no-relax
fi
exec "$python_command" -m pip install --no-build-isolation "$@" "$source_root"
