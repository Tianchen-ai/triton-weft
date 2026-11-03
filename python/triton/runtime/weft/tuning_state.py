"""Triton tuning keys and shared alias-preserving trial storage."""
import os

from weft_core.tuning_state import TrialCall, TuningState
from weft_core.buffer import Buffer


def workload_key(function, args, kwargs):
    supplied = function.signature.bind_partial(
        *args, **{name: value for name, value in kwargs.items() if name in function.arg_names})
    supplied.apply_defaults()
    constants = {parameter.name for parameter in function.params if parameter.is_constexpr}
    views = [(name, value if isinstance(value, Buffer) else Buffer(value))
             for name, value in supplied.arguments.items()
             if name not in constants and value is not None and not isinstance(value, (bool, int, float, str))]
    geometry = tuple((name, buffer.encoding, buffer.shape, buffer.strides, buffer.address % 256)
                     for name, buffer in views)
    spans = [buffer.byte_span() if buffer.encoding.startswith("dense.") else buffer.nbytes
             for _, buffer in views]
    aliases = []
    for index, (left_name, left) in enumerate(views):
        for right_index in range(index + 1, len(views)):
            right_name, right = views[right_index]
            if (spans[index] and spans[right_index]
                    and max(left.address, right.address) < min(left.address + spans[index], right.address + spans[right_index])):
                aliases.append((left_name, right_name, right.address - left.address))
    from triton.runtime.driver import driver
    target = driver.for_model(function.model).get_current_target()
    environment = tuple((key, repr(kwargs[key])) for key in sorted(kwargs)
                        if key not in function.arg_names and key not in ("grid", "warmup"))
    grid = kwargs.get("grid")
    launch = tuple(grid) if isinstance(grid, (tuple, list)) else None
    return (geometry, tuple(aliases), str(target), tuple(sorted(os.sched_getaffinity(0))),
            os.environ.get("OMP_NUM_THREADS"), environment, launch)
