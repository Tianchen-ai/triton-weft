"""Source semantics regressions independent of the benchmark harness."""
import numpy as np
import pytest
from semantic_programs import (semantics_composition, static_births,
    invalid_body_capture, invalid_sibling_capture, invalid_stage_update)
from semantic_reference import composition_reference


@pytest.mark.parametrize("shape", [(0, 0), (5, 0), (1, 1), (19, 37), (35, 129)])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("partition", [(17, 7), (4, 16)])
def test_snapshots_state_aliases_and_tails(shape, steps, partition):
    rows, block = partition
    rng = np.random.default_rng(19)
    value = rng.integers(-5000, 5001, shape, dtype=np.int32)
    initial = rng.integers(-32768, 32768, shape[0], dtype=np.int32)
    expected_input, expected, events = composition_reference(value, initial, steps, rows, block)
    for width in (1, 4, 8, 16):
        restricted_input, restricted, restricted_events = composition_reference(value, initial, steps, rows, block, width)
        np.testing.assert_array_equal(restricted_input, expected_input)
        np.testing.assert_array_equal(restricted, expected)
        assert restricted_events == events
    storage = np.full((shape[0], 5), -777, dtype=np.int32)
    output = storage[:, 1:4]
    semantics_composition[(1,)](value, initial, output, steps,
                               ROWS=rows, BLOCK=block, num_cpu_threads=1)
    np.testing.assert_array_equal(value, expected_input)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(storage[:, [0, 4]], -777)


@pytest.mark.parametrize("shared", [False, True])
def test_staged_observation_precedes_overwrite(shared):
    value = np.arange(7, dtype=np.float32)
    original = value.copy()
    output = np.empty_like(value)
    static_births[(1,)](value, output, SHARED=shared, num_cpu_threads=1)
    np.testing.assert_array_equal(value, original + 1)
    np.testing.assert_array_equal(output, original * (1 if shared else 2))


@pytest.mark.parametrize("function", [invalid_body_capture, invalid_sibling_capture, invalid_stage_update])
def test_invalid_initialization_and_snapshot_write_are_rejected(function):
    from triton.compiler.errors import CompilationError
    value = np.arange(7, dtype=np.float32)
    output = np.empty_like(value)
    with pytest.raises((CompilationError, ValueError)):
        function[(1,)](value, output, num_cpu_threads=1)
