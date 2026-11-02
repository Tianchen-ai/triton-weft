"""Normal Triton tuning over the existing pointer and mutable-alias programs."""
import triton

from .pointer_semantics import alias_shift, vector_add_tail


alias_shift_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK": 128}), triton.Config({"BLOCK": 256})],
    key=["N"],
)(alias_shift)

vector_add_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})],
    key=["N"],
)(vector_add_tail)
