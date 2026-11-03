"""Opaque in-process binding to the shared Canonical compiler C ABI."""
from __future__ import annotations

import ctypes as c
from functools import cache
import json
import os
from pathlib import Path
import weakref


class CompilationError(RuntimeError):
    def __init__(self, message):
        try:
            self.diagnostic = json.loads(message)
        except json.JSONDecodeError:
            self.diagnostic = {"message": message}
        self.report = self.diagnostic
        super().__init__(message)


class _Value(c.Structure):
    _fields_ = [("id", c.c_int64), ("type", c.c_char_p)]


class _Attribute(c.Structure):
    _fields_ = [("name", c.c_char_p), ("value", c.c_char_p)]


class _Operation(c.Structure):
    pass


class _Region(c.Structure):
    _fields_ = [("arguments", c.POINTER(_Value)), ("argument_count", c.c_size_t),
                ("operations", c.POINTER(_Operation)), ("operation_count", c.c_size_t)]


_Operation._fields_ = [
    ("name", c.c_char_p), ("filename", c.c_char_p), ("line", c.c_uint), ("column", c.c_uint),
    ("operands", c.POINTER(c.c_int64)), ("operand_count", c.c_size_t),
    ("results", c.POINTER(_Value)), ("result_count", c.c_size_t),
    ("attributes", c.POINTER(_Attribute)), ("attribute_count", c.c_size_t),
    ("regions", c.POINTER(_Region)), ("region_count", c.c_size_t),
]


@cache
def library():
    path = os.environ.get("WEFT_CORE_LIBRARY")
    if path is None:
        path = str(Path(__file__).with_name("lib") / "libweft_compiler.so")
    lib = c.CDLL(path, mode=os.RTLD_LOCAL)
    ptr, out = c.c_void_p, c.POINTER(c.c_void_p)
    signatures = {
        "context_create": [out, out], "context_release": [ptr],
        "module_build": [ptr, c.c_char_p, c.POINTER(_Operation), c.c_size_t, out, out],
        "module_parse": [ptr, c.c_char_p, out, out], "module_release": [ptr],
        "module_verify": [ptr, out], "module_text": [ptr, out, out],
        "module_location_snapshot": [ptr, c.c_char_p, out],
        "module_interface": [ptr, out, out],
        "module_lower": [ptr, c.c_char_p, out, out, out],
        "module_emit": [ptr, out, out],
        "target_info": [c.c_char_p, c.c_char_p, out, out],
        "string_release": [ptr],
    }
    for name, args in signatures.items():
        function = getattr(lib, "weft_compiler_" + name)
        function.argtypes = args
        function.restype = None if name.endswith("release") else c.c_int
    lib.weft_compiler_abi.argtypes = []
    lib.weft_compiler_abi.restype = c.c_uint
    if lib.weft_compiler_abi() != 1:
        raise RuntimeError("weft-core compiler ABI does not match this binding")
    return lib


def identity():
    """Cache identity for the installed schema and actual compiler build."""
    from . import __version__
    path = Path(library()._name).resolve()
    stat = path.stat()
    return {"version": __version__, "compiler_abi": 1, "canonical_schema": 1,
            "build": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}}


def _take(pointer):
    if not pointer.value:
        raise RuntimeError("compiler returned no diagnostic/result")
    try:
        return c.string_at(pointer).decode()
    finally:
        library().weft_compiler_string_release(pointer)


def _call(name, *arguments):
    error = c.c_void_p()
    status = getattr(library(), "weft_compiler_" + name)(*arguments, c.byref(error))
    if status:
        raise CompilationError(_take(error))


class Context:
    def __init__(self):
        self.handle = c.c_void_p()
        _call("context_create", c.byref(self.handle))
        self._release = weakref.finalize(self, library().weft_compiler_context_release, self.handle)


class Module:
    def __init__(self, context, handle):
        self.context = context
        self.handle = handle
        self._release = weakref.finalize(self, library().weft_compiler_module_release, handle)

    def verify(self):
        _call("module_verify", self.handle)
        return True

    def __str__(self):
        result = c.c_void_p()
        _call("module_text", self.handle, c.byref(result))
        return _take(result)

    def create_location_snapshot(self, filename):
        _call("module_location_snapshot", self.handle, os.fsencode(filename))


def parse_module(text, context=None):
    context = Context() if context is None else context
    result = c.c_void_p()
    _call("module_parse", context.handle, text.encode(), c.byref(result))
    return Module(context, result)


def create_module(context, name, operations, *, type_encoder):
    """Build the frontend's typed operations without serializing a module.

    Type and attribute spellings are local MLIR schema values; SSA identity,
    nested regions, operands and results cross as typed C records.
    """
    retained = []

    def string(value):
        data = value.encode()
        retained.append(data)
        return data

    def array(ctype, values):
        result = (ctype * len(values))(*values)
        retained.append(result)
        return result

    def value(item):
        return _Value(item.id, string(type_encoder(item.type)))

    def region(item):
        args = array(_Value, [value(arg) for arg in item.arguments])
        ops = array(_Operation, [operation(op) for op in item.operations])
        return _Region(args, len(args), ops, len(ops))

    def operation(item):
        operands = array(c.c_int64, [operand.id for operand in item.operands])
        results = array(_Value, [value(result) for result in item.results])
        attributes = array(_Attribute, [_Attribute(string(k), string(v)) for k, v in item.attributes.items()])
        regions = array(_Region, [region(r) for r in item.regions])
        loc = item.location
        return _Operation(string(item.name), string(loc.filename), loc.line, loc.column + 1,
                          operands, len(operands), results, len(results), attributes, len(attributes),
                          regions, len(regions))

    descriptions = array(_Operation, [operation(op) for op in operations])
    result = c.c_void_p()
    _call("module_build", context.handle, name.encode(), descriptions, len(descriptions), c.byref(result))
    return Module(context, result)


def interface(module):
    result = c.c_void_p()
    _call("module_interface", module.handle, c.byref(result))
    return json.loads(_take(result))


def lower(module, options):
    result, metadata = c.c_void_p(), c.c_void_p()
    _call("module_lower", module.handle, json.dumps(options).encode(), c.byref(result), c.byref(metadata))
    return Module(module.context, result), json.loads(_take(metadata))


def emit(module):
    result = c.c_void_p()
    _call("module_emit", module.handle, c.byref(result))
    return json.loads(_take(result))


def target_info(query, options):
    result = c.c_void_p()
    _call("target_info", query.encode(), json.dumps(options).encode(), c.byref(result))
    return json.loads(_take(result))


def native_target():
    return target_info("native_target", {})


def extension_catalog():
    return target_info("extension_catalog", {})


def x86_target_capabilities(march, abi, vlen_bits):
    return target_info("x86_target_capabilities", dict(march=march, abi=abi, vlen_bits=vlen_bits))


def target_capabilities(options):
    return target_info("target_capabilities", options)
