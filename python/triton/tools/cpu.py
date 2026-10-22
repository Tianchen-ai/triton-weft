"""Launch a Python program on an explicitly selected CPU group."""
import argparse
from dataclasses import asdict
import json
import os
import sys


def _cpus(text):
    values = set()
    for group in text.split(","):
        ends = group.split("-")
        if len(ends) == 1:
            values.add(int(ends[0]))
        elif len(ends) == 2 and int(ends[1]) >= int(ends[0]):
            values.update(range(int(ends[0]), int(ends[1]) + 1))
        else:
            raise ValueError("CPU selection must use IDs or ascending ranges")
    if not values or min(values) < 0:
        raise ValueError("CPU selection must contain nonnegative CPU IDs")
    return values


def main():
    from triton import cpu
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["native", *cpu.profiles()], default="native")
    parser.add_argument("--cpus")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--probe", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--inventory", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--register-ai", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("program", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.inventory:
        if args.register_ai:
            cpu._register_ai_thread()
        print(json.dumps(sorted(os.sched_getaffinity(0))))
        return
    if args.probe is not None:
        print(json.dumps(cpu._probe(args.probe, args.register_ai)))
        return
    if args.list:
        print(json.dumps(cpu.devices(), indent=2))
        return
    program = args.program[1:] if args.program[:1] == ["--"] else args.program
    if not program:
        parser.error("provide a Python script or use --list")
    if args.profile == "k3-a100":
        cpu._register_ai_thread()
    if args.cpus:
        selected = _cpus(args.cpus)
        cpu._bind_cpus(selected)
    from triton._C.libtriton import weft
    facts = weft.native_target()
    selected_target = cpu._validate_execution(args.profile, facts)
    print(json.dumps({"target": asdict(selected_target), "cpus": sorted(os.sched_getaffinity(0))}), file=sys.stderr)
    os.environ["TRITON_CPU_PROFILE"] = args.profile
    os.execv(sys.executable, [sys.executable, *program])


if __name__ == "__main__":
    main()
