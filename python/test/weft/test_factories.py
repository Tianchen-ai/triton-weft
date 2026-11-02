"""Ordinary Python factories preserve their captured program definitions."""
from __future__ import annotations

import triton
import triton.language as tl
from triton.runtime import _model_registry


def make_affine(encoding, factor):
    factor = tl.constexpr(factor)
    @triton.jit
    def calculate(value):
        return value * factor

    @triton.jit(model="weft")
    def affine(X: tl.View[encoding, (N,)], Y: tl.View[encoding, (N,)]):
        with tl.level.rows(N, group=8) as row:
            tl.store(Y[row], calculate(tl.load(X[row])))

    return affine


def test_factory_cache_keys_and_registry():
    first, second = make_affine(tl.f32, 2.0), make_affine(tl.f32, 3.0)
    repeated = make_affine(tl.f32, 2.0)
    other_type = make_affine(tl.f16, 2.0)
    assert first.fn.__qualname__ == second.fn.__qualname__
    assert first.cache_key != second.cache_key
    assert first.cache_key == repeated.cache_key
    assert first.cache_key != other_type.cache_key
    assert first.get_capture_scope()["encoding"] is tl.f32
    assert _model_registry.resolve(*_model_registry.reference(second)).cache_key == second.cache_key


def test_factory_execution_and_export(tmp_path):
    import numpy as np
    from triton.runtime.weft.artifact import load

    first, second = make_affine(tl.f32, 2.0), make_affine(tl.f32, 3.0)
    value = np.arange(19, dtype=np.float32)
    output = np.empty_like(value)
    for function, factor in [(first, 2.0), (second, 3.0), (first, 2.0)]:
        compiled = function[(1,)](value, output, num_cpu_threads=1)
        np.testing.assert_array_equal(output, value * factor)
    directory = tmp_path / "affine"
    compiled.export(directory)
    output.fill(-1)
    loaded = load(directory)
    loaded[(1,)](value, output)
    np.testing.assert_array_equal(output, value * 2.0)
