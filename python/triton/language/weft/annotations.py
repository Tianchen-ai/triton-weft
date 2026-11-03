"""Triton annotations over the shared Encoding and numerical schema."""
from weft_core.schema import (
    LayoutAtom, FieldLayoutSpec, PaddingSpec, ArraySpec, ViewSpec, ValueSpec,
    Value, static, AutoSpec, auto, grouped, bit_layers, pack_fields, padding,
    bitorder, byteorder, lo_first, hi_first,
)
from .dtypes import DType


class View:
    def __new__(cls, storage, *, encoding, shape, strides=None):
        from weft_core.buffer import Buffer
        from .declarations import EncodingDefinition, DerivedEncodingInstance
        from triton.compiler.weft.source import normalize_static
        encoding = normalize_static(encoding)
        if isinstance(encoding, str):
            identity = encoding
        elif isinstance(encoding, DType):
            identity = "dense." + encoding.name
        elif isinstance(encoding, EncodingDefinition):
            identity = encoding.__name__
        elif isinstance(encoding, DerivedEncodingInstance):
            identity = encoding.family.__name__ + "[" + ",".join(map(str, encoding.parameters)) + "]"
        else:
            raise TypeError("View encoding must be a dtype, declared Encoding or layout identity")
        return Buffer(storage, encoding=identity, shape=tuple(shape), strides=strides)

    @classmethod
    def __class_getitem__(cls, parameters):
        if not isinstance(parameters, tuple) or len(parameters) != 2:
            raise TypeError("View expects View[Encoding, shape]")
        encoding, shape = parameters
        return ViewSpec(encoding, shape if isinstance(shape, tuple) else (shape,))
