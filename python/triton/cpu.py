"""CPU compilation targets and explicit execution-device discovery.

Target construction is independent of execution. Device discovery uses isolated
processes, so probing a heterogeneous host never changes the caller's affinity.
"""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .backends.compiler import CPUTarget

_RVV = "rv64gcv_zfh_zvfh_zicbop_zihintpause"
_PROFILES = {
    "sg2044": CPUTarget("weft", "riscv64", _RVV, "lp64d", 128),
    "k1": CPUTarget("weft", "riscv64", _RVV + "_zba", "lp64d", 256, "spacemit-ime1"),
    "k3-x100": CPUTarget("weft", "riscv64", _RVV + "_zba", "lp64d", 256),
    "k3-a100": CPUTarget("weft", "riscv64", _RVV + "_zba", "lp64d", 1024, "spacemit-ime2"),
    "x86-avx2": CPUTarget("weft", "x86_64", "x86-64-v3", "sysv", 0),
    "x86-avx512": CPUTarget("weft", "x86_64", "x86-64-v4", "sysv", 0),
    "x86-amx": CPUTarget("weft", "x86_64", "x86-64-v4+amx-tile+amx-int8+amx-bf16", "sysv", 0),
}


def profiles():
    """Return documented compilation profiles; this does not probe hardware."""
    return dict(_PROFILES)


def _native_target(facts):
    native = facts["target"]
    architecture = native.get("arch", platform.machine())
    march = native["march"]
    extensions = facts.get("native_matrix_extensions", {})
    selected = "none"
    if architecture == "x86_64":
        # Native probing reports optional AMX separately. Admitting it here does
        # not acquire tile state; the actual worker checks used_extensions.
        present = set(march.split("+")[1:])
        if extensions:
            for name in ("amx-tile", *sorted(extensions)):
                if name not in present:
                    march += "+" + name
    else:
        execution_isa = facts["execution_isa_extensions"]
        present = set(march.split("_")[1:])
        for name in ("zifencei", "zicbop"):
            if name in execution_isa and name not in present:
                march += "_" + name
        allowed = set(facts["cpus"])
        supported = sorted(name for name, cpus in extensions.items() if allowed <= set(cpus))
        if len(supported) > 1:
            raise ValueError("multiple matrix extensions need an explicit compilation target")
        if supported:
            selected = supported[0]
    return CPUTarget("weft", architecture, march, native["abi"], native["vlen_bits"], selected)


def target(profile="native"):
    """Construct a target for JIT/AOT. Named profiles also work on cross hosts."""
    if profile == "native":
        from ._C.libtriton import weft
        return _native_target(weft.native_target())
    if profile not in _PROFILES:
        raise ValueError(f"unknown CPU profile {profile!r}; choose native or {', '.join(_PROFILES)}")
    return _PROFILES[profile]


def _execution_target(facts, profile=None):
    """Apply the launch profile to discovered facts without changing GPU state."""
    if profile is None or profile == "native":
        return _native_target(facts)
    return _validate_execution(profile, facts)


def _register_ai_thread():
    path = Path("/proc/set_ai_thread")
    if not path.exists():
        raise RuntimeError("this CPU profile requires the OS /proc/set_ai_thread interface")
    path.write_text(str(os.getpid()))


def _probe(cpu, register):
    if register:
        _register_ai_thread()
    _bind_cpus({cpu})
    from ._C.libtriton import weft
    return weft.native_target()


def _bind_cpus(selected):
    os.sched_setaffinity(0, selected)
    observed = os.sched_getaffinity(0)
    if observed != selected:
        raise RuntimeError(f"OS did not retain requested CPU set {sorted(selected)}; allowed set is {sorted(observed)}")


def devices():
    """Return homogeneous CPU groups and observed startup requirements.

    Unavailable probes retain their diagnostics. A reported matrix capability
    is not evidence that a program has executed matrix instructions.
    """
    from dataclasses import asdict
    groups = {}
    unavailable = []
    registration = platform.machine() == "riscv64" and Path("/proc/set_ai_thread").exists()
    current = set(os.sched_getaffinity(0))
    candidates = set(current)
    if registration:
        result = subprocess.run([sys.executable, "-m", "triton.tools.cpu", "--inventory", "--register-ai"],
                                text=True, capture_output=True)
        if result.returncode:
            unavailable.append({"startup": "register-ai-before-affinity", "error": result.stderr.strip(),
                                "exit_code": result.returncode})
        else:
            candidates.update(json.loads(result.stdout))
    for cpu in sorted(candidates):
        observations = []
        attempts = (False, True) if registration and cpu in current else (True,) if registration else (False,)
        for register in attempts:
            result = subprocess.run([sys.executable, "-m", "triton.tools.cpu", "--probe", str(cpu),
                                     *( ["--register-ai"] if register else [])], text=True, capture_output=True)
            if result.returncode:
                observations.append({"registration": register, "exit_code": result.returncode,
                                     "error": result.stderr.strip()})
                continue
            facts = json.loads(result.stdout)
            descriptor = asdict(_native_target(facts))
            key = json.dumps(descriptor, sort_keys=True)
            group = groups.setdefault(key, {"target": descriptor, "cpus": [],
                                           "startup": "register-ai-before-affinity" if register else "affinity"})
            group["cpus"].append(cpu)
            break
        else:
            unavailable.append({"cpu": cpu, "observations": observations})
    return {"current_cpus": sorted(current), "groups": list(groups.values()), "unavailable": unavailable}


def _validate_execution(profile, facts):
    from .backends.weft.target import target_capabilities
    expected = target(profile)
    observed = _native_target(facts)
    if (expected.arch, expected.abi, expected.vlen_bits) != (observed.arch, observed.abi, observed.vlen_bits):
        raise RuntimeError(f"selected CPUs do not match profile {profile}: {observed}")
    if expected.matrix_extension != "none" and expected.matrix_extension != observed.matrix_extension:
        raise RuntimeError(f"selected CPUs do not provide {expected.matrix_extension}")
    wanted = target_capabilities(expected, matrix_extension=expected.matrix_extension)
    actual = target_capabilities(observed, matrix_extension=observed.matrix_extension)
    wanted_isa = wanted["isa_extensions"]
    if observed.arch == "riscv64":
        actual_isa = facts["execution_isa_extensions"]
        missing = [name for name, version in wanted_isa.items()
                   if name not in actual_isa or tuple(actual_isa[name]) < tuple(version)]
    else:
        missing = set(wanted_isa) - set(actual["isa_extensions"])
    if missing:
        raise RuntimeError(f"selected CPUs lack target ISA features or versions: {sorted(missing)}")
    return observed
