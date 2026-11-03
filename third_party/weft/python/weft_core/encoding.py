"""Serializable Encoding declarations shared by native source frontends."""
from dataclasses import asdict, dataclass
import json
import math

from . import dtypes
from .schema import ArraySpec, FieldLayoutSpec
from .types import ScalarType, emit_type


def integer_array(values):
    return "array<i64: " + ", ".join(map(str, values)) + ">" if values else "array<i64>"


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str
    shape: tuple
    layouts: tuple
    bit_offset: int
    storage_bits: int


@dataclass(frozen=True)
class Encoding:
    name: str
    bit_order: str
    byte_order: str
    alignment: int
    elements: int
    fields: tuple[Field, ...]
    padding: tuple
    storage_bits: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["fields"] = tuple(Field(**{**field, "shape": tuple(field["shape"]),
                                          "layouts": tuple(field["layouts"])}) for field in values["fields"])
        values["padding"] = tuple(tuple(pad) for pad in values["padding"])
        return cls(**values)

    def attributes(self):
        layouts = []
        for field in self.fields:
            atoms = []
            for atom in field.layouts:
                values = ["kind = " + json.dumps(atom["kind"])]
                for key in ("size", "fields", "low_bits"):
                    if atom.get(key):
                        values.append(f"{key} = {atom[key]} : i64")
                if atom.get("order"):
                    values.append("order = " + json.dumps(atom["order"]))
                if atom.get("role", -1) >= 0:
                    values.append(f"role = {atom['role']} : i64")
                atoms.append("{" + ", ".join(values) + "}")
            layouts.append("[" + ", ".join(atoms) + "]")
        return {
            "sym_name": json.dumps(self.name), "kind": json.dumps("base"),
            "layout_identity": json.dumps(self.name), "bit_order": json.dumps(self.bit_order),
            "byte_order": json.dumps(self.byte_order), "alignment": str(self.alignment),
            "elements": str(self.elements), "storage_bits": str(self.storage_bits),
            "field_names": json.dumps([f.name for f in self.fields]),
            "field_types": "[" + ", ".join(emit_type(ScalarType(getattr(dtypes, f.dtype))) for f in self.fields) + "]",
            "field_shapes": "[" + ", ".join(integer_array(f.shape) for f in self.fields) + "]",
            "field_layouts": "[" + ", ".join(layouts) + "]",
            "field_bit_offsets": integer_array([f.bit_offset for f in self.fields]),
            "field_storage_bits": integer_array([f.storage_bits for f in self.fields]),
            "padding": integer_array([component for pad in self.padding for component in pad]),
        }


def storage_declaration(name, *, layout, alignment, elements, items):
    orders = {atom.category: atom.value for atom in layout}
    if len(layout) != 2 or set(orders) != {"bit_order", "byte_order"}:
        raise ValueError("Encoding layout needs exactly one bit order and byte order")
    if type(elements) is not int or elements <= 0 or type(alignment) is not int or alignment <= 0:
        raise ValueError("Encoding elements and byte alignment must be positive integers")
    fields, padding, offset = [], [], 0
    cursor = 0
    while cursor < len(items):
        field_name, spec = items[cursor]
        if not isinstance(spec, ArraySpec):
            if offset % 8:
                raise ValueError("byte padding must start at a byte boundary")
            padding.append((offset, spec.bytes * 8, spec.value))
            offset += spec.bytes * 8
            cursor += 1
            continue
        if spec.dtype.bits is None or any(type(n) is not int or n <= 0 for n in spec.shape):
            raise ValueError("Encoding fields require a fixed-width dtype and positive static shape")
        layouts = spec.layouts or (FieldLayoutSpec("natural", 0),)
        if layouts[0].kind == "joined":
            group = layouts[0]
            if len(layouts) != 1 or group.fields < 2 or len(spec.shape) != 1 or spec.shape[0] != 2 * group.size:
                raise ValueError("joined fields require one complete consecutive field group")
            width = group.size * (group.fields + 1) * 8
            for role in range(group.fields):
                if cursor + role >= len(items) or items[cursor + role][1] != spec:
                    raise ValueError("pack_fields members must have identical dtype, shape and layout")
                atom = {**asdict(group), "role": role}
                fields.append(Field(items[cursor + role][0], spec.dtype.name, spec.shape, (atom,), offset, width))
            cursor += group.fields
        else:
            width = spec.dtype.bits * math.prod(spec.shape)
            fields.append(Field(field_name, spec.dtype.name, spec.shape, tuple(asdict(atom) for atom in layouts), offset, width))
            cursor += 1
        offset += width
    return Encoding(name, orders["bit_order"], orders["byte_order"], alignment, elements,
                    tuple(fields), tuple(padding), offset)
