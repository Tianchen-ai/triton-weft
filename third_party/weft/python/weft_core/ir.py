from __future__ import annotations

import json
from dataclasses import dataclass, field

from .diagnostics import SourceLocation




@dataclass(frozen=True, slots=True)
class Value:
    id: int
    type: object
    name_hint: str | None = None
    builder: object = field(default=None, compare=False, hash=False, repr=False)
    location: object = field(default=None, compare=False, hash=False, repr=False)
    producer: object = field(default=None, compare=False, hash=False, repr=False)

    @property
    def name(self) -> str:
        return f"%v{self.id}"

    def get_type(self):
        return self.type

    def get_loc(self):
        return self.location

    def set_loc(self, location):
        object.__setattr__(self, "location", location)

    def replace_all_uses_with(self, replacement):
        for region in self.builder.regions:
            for operation in region.operations:
                operation.operands = tuple(replacement if value is self else value
                                           for value in operation.operands)


@dataclass(slots=True)
class Region:
    arguments: tuple[Value, ...]
    operations: list[Operation] = field(default_factory=list)

    def arg(self, index):
        return self.arguments[index]

    def has_terminator(self):
        return self.terminated

    def erase(self):
        self.operations.clear()

    def merge_block_before(self, destination):
        if self.arguments:
            raise ValueError("cannot merge an argument-bearing canonical block")
        destination.operations[:0] = self.operations
        self.operations.clear()

    def get_parent(self):
        return self

    def size(self):
        return 1

    def used_values(self) -> set[Value]:
        values = set()
        for operation in self.operations:
            values.update(operation.operands)
            for region in operation.regions:
                values.update(region.used_values())
        return values

    @property
    def terminated(self) -> bool:
        return bool(self.operations) and self.operations[-1].name in {
            "weft_kernel.return",
            "scf.yield",
            "scf.condition",
            "weft_kernel.births_yield",
            "weft_kernel.handoff",
            "weft_kernel.derive_yield",
            "weft_kernel.reduce_yield",
            "weft_kernel.scan_yield",
        }


@dataclass(slots=True)
class Operation:
    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: dict[str, str]
    regions: tuple[Region, ...]
    location: SourceLocation

    def get_result(self, index):
        return self.results[index]

    def get_num_results(self):
        return len(self.results)

    def get_then_block(self):
        return self.regions[0]

    def get_else_block(self):
        return self.regions[1]

    def get_body(self, index=0):
        return self.regions[index]

    def get_region(self, index):
        return self.regions[index]

    def verify(self):
        # Native construction validates the complete Canonical operation after
        # the shared frontend has populated its combiner and SSA operands.
        return all(region.terminated for region in self.regions)

    def get_before(self):
        return self.regions[0]

    def get_after(self):
        return self.regions[1]

    def get_induction_var(self):
        return self.regions[0].arguments[0]

    def set_attr(self, name, value):
        self.attributes[name] = value


class IRBuilder:
    """One-way canonical MLIR assembly builder."""

    def __init__(self) -> None:
        self._next_value_id = 0
        self.regions = []

    def value(self, value_type: object, name_hint: str | None = None) -> Value:
        value = Value(self._next_value_id, value_type, name_hint, self)
        self._next_value_id += 1
        return value

    def region(
        self,
        argument_types: tuple[object, ...],
        argument_names: tuple[str | None, ...] = (),
    ) -> Region:
        names = argument_names or (None,) * len(argument_types)
        if len(names) != len(argument_types):
            raise ValueError("region argument names and types must match")
        region = Region(
            tuple(
                self.value(value_type, name)
                for value_type, name in zip(argument_types, names)
            )
        )
        self.regions.append(region)
        return region

    def operation(
        self,
        name: str,
        location: SourceLocation,
        *,
        operands: tuple[Value, ...] = (),
        result_types: tuple[object, ...] = (),
        attributes: dict[str, str] | None = None,
        regions: tuple[Region, ...] = (),
        result_names: tuple[str | None, ...] = (),
    ) -> Operation:
        names = result_names or (None,) * len(result_types)
        if len(names) != len(result_types):
            raise ValueError("result names and types must match")
        results = tuple(
            self.value(value_type, hint)
            for value_type, hint in zip(result_types, names)
        )
        operation = Operation(
            name,
            tuple(operands),
            results,
            dict(attributes or {}),
            tuple(regions),
            location,
        )
        for result in results:
            object.__setattr__(result, "producer", operation)
        return operation

    def emit(
        self,
        block: Region,
        name: str,
        location: SourceLocation,
        **kwargs: object,
    ) -> tuple[Value, ...]:
        if block.terminated:
            raise ValueError("cannot emit after a block terminator")
        operation = self.operation(name, location, **kwargs)
        block.operations.append(operation)
        return operation.results

