from dataclasses import asdict, dataclass, field, fields
import hashlib
import json

from triton._C.libtriton import weft
from triton.backends.compiler import BaseBackend, CPUTarget, Language


@dataclass(frozen=True)
class CPUOptions:
    # Shared builtin semantics, independent of CPU scheduling and physical tuning.
    allowed_dot_input_precisions = ("ieee",)
    default_dot_input_precision = "ieee"
    sanitize_overflow = False
    meta: dict = field(default_factory=dict)
    num_cpu_threads: int = 0
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
    unroll: int = 1
    pipeline_depth: int = 1
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
    matrix_extension: str = "none"
    required_extensions: tuple = ()
    local_tile_strategy: str = "target-default"
    private_stack_bytes: int = 65536
    data_cache: dict | None = None


class WeftBackend(BaseBackend):
    binary_ext = "so"
    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target):
        return isinstance(target, CPUTarget) and target.backend == "weft" and target.arch in ("riscv64", "x86_64")

    def hash(self):
        return str(self.target)

    @staticmethod
    def merge_options(options, source_options):
        values = BaseBackend.merge_options(options, source_options)
        meta = dict(options.get("meta", {}))
        physical = dict(options.get("backend_options", {}))
        if "meta" in physical:
            meta.update(physical.pop("meta"))
            values["backend_options"] = physical
        for name, value in source_options.get("meta", {}).items():
            if name in meta and meta[name] != value:
                raise ValueError(f"source parameter {name!r} conflicts with constexpr value {value}")
            meta[name] = value
        values["meta"] = meta
        return values

    def parse_options(self, options):
        from .toolchain import Toolchain
        values = dict(options)
        physical = dict(values.pop("backend_options", {}))
        option_class = X86Options if self.target.arch == "x86_64" else WeftOptions
        valid = {f.name for f in fields(option_class)}
        if set(physical) - valid:
            raise ValueError(f"unknown Weft backend options: {sorted(set(physical) - valid)}")
        if values.keys() - valid:
            raise ValueError(f"unknown Weft compilation options: {sorted(values.keys() - valid)}")
        values.update(physical)
        if self.target.arch == "x86_64":
            weft.x86_target_capabilities(self.target.march, self.target.abi, self.target.vlen_bits)
        values["toolchain"] = Toolchain.resolve(values.get("toolchain", {}), self.target).to_dict()
        result = option_class(**values)
        if self.target.arch == "x86_64" and type(result.tune_cpu) is not str:
            raise ValueError("tune_cpu must be an LLVM x86 CPU name or an empty string")
        if type(result.num_cpu_threads) is not int or result.num_cpu_threads < 0:
            raise ValueError("num_cpu_threads must be a nonnegative integer; 0 uses the OpenMP setting")
        if result.instrumentation_mode:
            raise NotImplementedError("GPU instrumentation does not implement Weft CPU effects")
        return result

    def load_dialects(self, context):
        weft.load_dialects(context)

    def get_module_map(self):
        from triton.language.weft import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def get_codegen_implementation(self, options):
        # Canonical contraction has no GPU tile-size floor. Physical legality
        # remains the responsibility of the selected target implementation.
        return {"min_dot_size": lambda lhs, rhs: (1, 1, 1)}

    def pack_metadata(self, metadata):
        return metadata

    @staticmethod
    def specialize_argument(backend, value, is_const, specialize, align):
        from triton.runtime.weft.buffer import storage_encoding
        if isinstance(value, bool):
            return ("u1", "")
        if isinstance(value, int):
            return ("i64", "")
        if isinstance(value, float):
            return ("fp32", "")
        encoding = storage_encoding(value)
        if encoding.startswith("dense."):
            from triton.compiler.weft.values import logical_dtype
            from triton.language.weft import dtypes
            from triton.runtime.jit import _normalize_ty
            return ("*" + _normalize_ty(logical_dtype(getattr(dtypes, encoding.removeprefix("dense.")))), "")
        return ("view:" + encoding, "")

    @staticmethod
    def parse_attr(description):
        return []

    def add_stages(self, stages, options, language):
        if language != Language.WEFT:
            raise ValueError("the Weft backend requires @triton.jit(model='weft')")

        def canonical(module, metadata):
            if not module.verify():
                raise ValueError("invalid Weft canonical IR")
            metadata["interface"] = weft.interface(module)
            return module

        def physical(module, metadata):
            bindings = asdict(options)
            interface = metadata["interface"]["interface"]
            names = set(interface["source_parameters"]) | set(interface["shape_parameters"])
            bindings["meta"] = {name: value for name, value in options.meta.items() if name in names}
            bindings.update(march=self.target.march, abi=self.target.abi, vlen_bits=self.target.vlen_bits)
            lower = weft.lower_x86 if self.target.arch == "x86_64" else weft.lower
            lowered, report = lower(module, bindings)
            lowered.context = module.context
            metadata["compilation"] = report
            metadata["source_bindings"] = bindings["meta"]
            return lowered

        def emit(module, metadata):
            x86 = self.target.arch == "x86_64"
            result = weft.emit_x86(module) if x86 else weft.emit(module)
            if x86:
                result["kernels"] = [result["kernel"]]
            if len(result["kernels"]) != 1:
                raise ValueError("a Triton JITFunction must produce exactly one native kernel")
            metadata["kernel_abi"] = result["kernels"][0]
            from .launcher import single_instance_abi
            metadata["single_instance_abi"] = single_instance_abi(metadata["kernel_abi"])
            metadata["name"] = metadata["kernel_abi"]["symbol"]
            metadata["compilation"]["pipeline"].extend(result["compilation"]["pipeline"])
            return result["llvm_ir"] if x86 else result["intrinsic_c"]

        def native(source, metadata):
            from .toolchain import Toolchain
            return Toolchain(**options.toolchain).compile(source, metadata, self.target)

        stages["weftir"] = canonical
        stages["x86ir" if self.target.arch == "x86_64" else "riscvir"] = physical
        stages["llir" if self.target.arch == "x86_64" else "c"] = emit
        stages["so"] = native
