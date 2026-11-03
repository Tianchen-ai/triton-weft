"""Finite physical configuration shared by embedding frontends."""
from dataclasses import asdict, dataclass, field
import hashlib
import json


@dataclass(frozen=True)
class CPUOptions:
    allowed_dot_input_precisions = ("ieee",)
    default_dot_input_precision = "ieee"
    sanitize_overflow = False
    meta: dict = field(default_factory=dict)
    num_cpu_threads: int = 0
    pipeline_depth: int = 1
    toolchain: dict = field(default_factory=dict)
    debug: bool = False
    instrumentation_mode: str = ""

    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class X86Options(CPUOptions):
    tune_cpu: str = ""


@dataclass(frozen=True)
class WeftOptions(CPUOptions):
    lmul_eighths: int = 8
    native_loop_alignment: int = 32
    unroll: int = 1
    scalar_load_prime: int = 0
    pointwise_issue_copies: int = 1
    contraction_column_copies: int = 4
    panel_transfer_fields: int = 4
    panel_transfer_strategy: str = "issue-work"
    contraction_panel_k: int = 128
    contraction_k_steps: int = 4
    partial_combine_policy: str = "independent-multilevel"
    record_axis_policy: str = "within-record"
    max_widening_combine_groups: int = 2
    matrix_extension: str = "auto"
    required_extensions: tuple = ()
    local_tile_strategy: str = "target-default"
    private_stack_bytes: int = 65536
    data_cache: dict | None = None
