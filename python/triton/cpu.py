"""CPU targets and execution devices from the shared Weft SDK."""
from weft_core.cpu import (
    profiles, target, devices, _execution_target, _native_target,
    _register_ai_thread, _probe, _bind_cpus, _validate_execution,
)

__all__ = ["profiles", "target", "devices"]
