"""GGML record layouts shared by embedded source languages.

These are byte/field declarations, not dequantizers or dispatch templates.
Algorithms remain in the frontend's ordinary kernel library.
"""
from types import MappingProxyType
from ..dtypes import f16, f32, i8, i16, u1, u2, u4, u6, u8, u16, u32
from ..encoding import storage_declaration
from ..schema import ArraySpec, bitorder, byteorder, grouped, bit_layers, pack_fields, lo_first


def _schema(name, elements, fields, alignment=1):
    return storage_declaration(name, layout=(bitorder.lsb_first, byteorder.little),
                               elements=elements, alignment=alignment, items=fields)


def _scalar(dtype):
    return ArraySpec(dtype, ())


def _layers(dtype, count, group, layer):
    return dtype[count] @ grouped(elements=group) @ bit_layers(elements=layer, order=lo_first)


Q1_0 = _schema("Q1_0", 128, [("d", _scalar(f16)), ("q", _layers(u1, 128, 8, 1))], 2)
Q4_0 = _schema("Q4_0", 32, [("d", _scalar(f16)), ("q", _layers(u4, 32, 32, 16))], 2)
Q4_1 = _schema("Q4_1", 32, [("d", _scalar(f16)), ("m", _scalar(f16)), ("q", _layers(u4, 32, 32, 16))], 2)
Q5_0 = _schema("Q5_0", 32, [("d", _scalar(f16)), ("qh", _layers(u1, 32, 8, 1)),
                           ("q", _layers(u4, 32, 32, 16))], 2)
Q5_1 = _schema("Q5_1", 32, [("d", _scalar(f16)), ("m", _scalar(f16)),
                           ("qh", _layers(u1, 32, 8, 1)), ("q", _layers(u4, 32, 32, 16))], 2)
Q8_0 = _schema("Q8_0", 32, [("d", _scalar(f16)), ("q", i8[32])], 2)
Q8_1 = _schema("Q8_1", 32, [("d", _scalar(f16)), ("s", _scalar(f16)), ("q", i8[32])], 2)
Q2_K = _schema("Q2_K", 256, [("scales", u8[16]), ("q", _layers(u2, 256, 128, 32)),
                            ("d", _scalar(f16)), ("dmin", _scalar(f16))], 2)
Q3_K = _schema("Q3_K", 256, [("hmask", _layers(u1, 256, 256, 32)),
                            ("q", _layers(u2, 256, 128, 32)), ("scale_low", _layers(u4, 16, 16, 8)),
                            ("scale_high", _layers(u2, 16, 16, 4)), ("d", _scalar(f16))], 2)
_scale_min = u6[8] @ pack_fields(group=4, fields=2, low_bits=4, order=lo_first)
Q4_K = _schema("Q4_K", 256, [("d", _scalar(f16)), ("dmin", _scalar(f16)),
                            ("sc", _scale_min), ("m", _scale_min), ("q", _layers(u4, 256, 64, 32))], 2)
Q5_K = _schema("Q5_K", 256, [("d", _scalar(f16)), ("dmin", _scalar(f16)),
                            ("sc", _scale_min), ("m", _scale_min), ("qh", _layers(u1, 256, 256, 32)),
                            ("q", _layers(u4, 256, 64, 32))], 2)
Q6_K = _schema("Q6_K", 256, [("ql", _layers(u4, 256, 128, 64)), ("qh", _layers(u2, 256, 128, 32)),
                            ("scales", _layers(i8, 16, 16, 16)), ("d", _scalar(f16))])
Q8_K = _schema("Q8_K", 256, [("ds", _scalar(f32)), ("q", i8[256]), ("bsum", i16[16])], 4)
IQ1_S = _schema("IQ1_S", 256, [("d", _scalar(f16)), ("q", u8[32]), ("qh", u16[8])], 2)
IQ1_M = _schema("IQ1_M", 256, [("q", u8[32]), ("qh", u8[16]), ("scales", u8[8])])
IQ2_S = _schema("IQ2_S", 256, [("d", _scalar(f16)), ("q", u8[32]),
                              ("signs", _layers(u1, 256, 8, 1)), ("qh", u8[8]), ("scales", u8[8])], 2)
IQ2_XS = _schema("IQ2_XS", 256, [("d", _scalar(f16)), ("q", u16[32]), ("scales", u8[8])], 2)
IQ2_XXS = _schema("IQ2_XXS", 256, [("d", _scalar(f16)), ("q", u16[32])], 2)
I8X8 = _schema("I8X8", 8, [("values", i8[8])], 8)
I8X4 = _schema("I8X4", 4, [("values", i8[4])], 4)
IQ3_S = _schema("IQ3_S", 256, [("d", _scalar(f16)), ("q", u8[64]), ("qh", _layers(u1, 64, 8, 1)),
                              ("signs", _layers(u1, 256, 8, 1)), ("scales", _layers(u4, 8, 2, 1))], 2)
IQ3_XXS = _schema("IQ3_XXS", 256, [("d", _scalar(f16)), ("q", u8[64]), ("metadata", u32[8])], 2)
IQ4_NL = _schema("IQ4_NL", 32, [("d", _scalar(f16)), ("q", _layers(u4, 32, 32, 16))], 2)
IQ4_XS = _schema("IQ4_XS", 256, [("d", _scalar(f16)), ("scales_h", _scalar(u16)),
                                ("scales_l", u8[4]), ("q", _layers(u4, 256, 32, 16))], 2)
TQ1_0 = _schema("TQ1_0", 256, [("q", u8[48]), ("qh", u8[4]), ("d", _scalar(f16))], 2)
TQ2_0 = _schema("TQ2_0", 256, [("q", _layers(u2, 256, 128, 32)), ("d", _scalar(f16))], 2)
MXFP4 = _schema("MXFP4", 32, [("e", _scalar(u8)), ("q", _layers(u4, 32, 32, 16))])
NVFP4 = _schema("NVFP4", 64, [("d", u8[4]), ("q", _layers(u4, 64, 16, 8))])

CATALOG = MappingProxyType({schema.name: schema for schema in (
    Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
    IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, I8X8, I8X4, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS,
    TQ1_0, TQ2_0, MXFP4, NVFP4,
)})
