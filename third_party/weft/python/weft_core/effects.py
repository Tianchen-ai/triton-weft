from __future__ import annotations
import json
from .ir import Value, Region, Operation, IRBuilder
from .types import ViewType, SliceType, PointerType


def argument_accesses(body: Region) -> list[set[str]]:
    """Infer ABI memory effects through the completed structured SSA graph.

    This also covers the common frontend's speculative carry discovery: only
    regions in the final program contribute effects. View joins may have more
    than one possible origin; every possible input retains the access effect.
    """
    roots = {value: {index} for index, value in enumerate(body.arguments)
             if isinstance(value.type, (ViewType, PointerType))}
    edges = []
    memory = []

    def connect(destinations, sources):
        for destination, source in zip(destinations, sources):
            if isinstance(destination.type, (ViewType, SliceType, PointerType)):
                edges.append((destination, source))

    def walk(block):
        for operation in block.operations:
            name, args, results, regions = operation.name, operation.operands, operation.results, operation.regions
            if name in ("weft_kernel.slice", "weft_kernel.subview", "weft_kernel.field", "weft_kernel.pointer_cast"):
                connect(results, args[:1])
            elif name == "scf.if":
                for region in regions:
                    connect(results, region.operations[-1].operands)
            elif name == "scf.for":
                carried = regions[0].arguments[1:]
                connect(carried, args[3:])
                connect(carried, regions[0].operations[-1].operands)
                connect(results, carried)
            elif name == "scf.while":
                before, after = regions
                connect(before.arguments, args)
                connect(before.arguments, after.operations[-1].operands)
                forwarded = before.operations[-1].operands[1:]
                connect(after.arguments, forwarded)
                connect(results, forwarded)
            elif name == "weft_kernel.helper":
                def return_edges(region):
                    for inner in region.operations:
                        if inner.name == "weft_kernel.helper_return":
                            connect(results, inner.operands)
                        elif inner.name != "weft_kernel.helper":
                            for nested in inner.regions:
                                return_edges(nested)
                return_edges(regions[0])
            elif name == "weft_kernel.level":
                state, staged, loop = regions
                carried = loop.arguments[1:1 + len(results)]
                connect(carried, args[1:])
                connect(carried, loop.operations[-1].operands)
                connect(results, carried)
                connect(loop.arguments[1 + len(results):],
                        state.operations[-1].operands + staged.operations[-1].operands)
            if name == "weft_kernel.admit":
                memory.append((args[0], ("read",)))
            elif name in ("weft_kernel.pointer_load", "weft_kernel.pointer_store"):
                memory.append((args[0], ("read" if name.endswith("load") else "write",)))
            elif name == "weft_kernel.pointer_atomic":
                memory.append((args[0], ("read", "write")))
            elif name == "weft_kernel.commit":
                memory.append((args[1], ("write",)))
            elif name == "weft_kernel.atomic_rmw":
                memory.append((args[1], ("read", "write")))
            for region in regions:
                walk(region)

    walk(body)
    changed = True
    while changed:
        changed = False
        for destination, source in edges:
            known = roots.setdefault(destination, set())
            incoming = roots.get(source, set())
            if not incoming.issubset(known):
                known.update(incoming)
                changed = True
    accesses = [set() for _ in body.arguments]
    for value, effects in memory:
        for index in roots.get(value, ()):
            accesses[index].update(effects)
    return accesses
