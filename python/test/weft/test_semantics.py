"""Source semantics regressions independent of the benchmark harness."""
import numpy as np
import pytest


def test_cpu_selection_registers_before_binding(monkeypatch):
    from weft_core import cpu
    observed, calls = {0, 1}, []
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda _: observed.copy())
    monkeypatch.setattr(cpu, "devices", lambda: {"groups": [
        {"cpus": [8, 9], "startup": "register-ai-before-affinity"}], "unavailable": []})

    def register():
        calls.append("register")
        observed.update({8, 9})

    def bind(_, cpus):
        assert cpus <= observed
        calls.append(("bind", cpus))
        observed.intersection_update(cpus)

    monkeypatch.setattr(cpu, "_register_ai_thread", register)
    monkeypatch.setattr(cpu.os, "sched_setaffinity", bind)
    cpu.bind_cpus([8, 9])
    assert calls == ["register", ("bind", {8, 9})]
    assert observed == {8, 9}


def test_cpu_selection_rejects_mixed_or_missing_group(monkeypatch):
    from weft_core import cpu
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda _: {0, 1})
    monkeypatch.setattr(cpu, "devices", lambda: {"groups": [
        {"cpus": [0, 1], "startup": "affinity"},
        {"cpus": [8, 9], "startup": "register-ai-before-affinity"}], "unavailable": []})
    monkeypatch.setattr(cpu, "_register_ai_thread", lambda: pytest.fail("must reject before startup"))
    monkeypatch.setattr(cpu.os, "sched_setaffinity", lambda *_: pytest.fail("must reject before binding"))
    with pytest.raises(RuntimeError, match="homogeneous"):
        cpu.bind_cpus([0, 8])
    with pytest.raises(RuntimeError, match="homogeneous"):
        cpu.bind_cpus([10])


def test_cpu_selection_keeps_available_affinity_local(monkeypatch):
    from weft_core import cpu
    selected = {0, 1}
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda _: selected.copy())
    monkeypatch.setattr(cpu.os, "sched_setaffinity", lambda _, cpus: selected.intersection_update(cpus))
    monkeypatch.setattr(cpu, "devices", lambda: pytest.fail("an available selection needs no whole-machine probe"))
    cpu.bind_cpus([1])
    assert selected == {1}
from semantic_programs import (semantics_composition, static_births,
    invalid_body_capture, invalid_sibling_capture, invalid_stage_update,
    dense_index_conversion)
from semantic_reference import composition_reference


@pytest.mark.parametrize("shape", [(0, 0), (5, 0), (1, 1), (19, 37), (35, 129)])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("partition", [(17, 7), (4, 16)])
def test_snapshots_state_aliases_and_tails(shape, steps, partition):
    rows, block = partition
    rng = np.random.default_rng(19)
    value = rng.integers(-5000, 5001, shape, dtype=np.int32)
    initial = rng.integers(-32768, 32768, shape[0], dtype=np.int32)
    expected_input, expected, events = composition_reference(value, initial, steps, rows, block)
    for width in (1, 4, 8, 16):
        restricted_input, restricted, restricted_events = composition_reference(value, initial, steps, rows, block, width)
        np.testing.assert_array_equal(restricted_input, expected_input)
        np.testing.assert_array_equal(restricted, expected)
        assert restricted_events == events
    storage = np.full((shape[0], 5), -777, dtype=np.int32)
    output = storage[:, 1:4]
    semantics_composition[(1,)](value, initial, output, steps,
                               ROWS=rows, BLOCK=block, num_cpu_threads=1)
    np.testing.assert_array_equal(value, expected_input)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(storage[:, [0, 4]], -777)


@pytest.mark.parametrize("shared", [False, True])
def test_staged_observation_precedes_overwrite(shared):
    value = np.arange(7, dtype=np.float32)
    original = value.copy()
    output = np.empty_like(value)
    static_births[(1,)](value, output, SHARED=shared, num_cpu_threads=1)
    np.testing.assert_array_equal(value, original + 1)
    np.testing.assert_array_equal(output, original * (1 if shared else 2))


@pytest.mark.parametrize("function", [invalid_body_capture, invalid_sibling_capture, invalid_stage_update])
def test_invalid_initialization_and_snapshot_write_are_rejected(function):
    from triton.compiler.errors import CompilationError
    value = np.arange(7, dtype=np.float32)
    output = np.empty_like(value)
    with pytest.raises((CompilationError, ValueError)):
        function[(1,)](value, output, num_cpu_threads=1)


@pytest.mark.parametrize("size", [1, 64, 65, 137])
def test_dense_index_conversion_preserves_tail(size):
    output = np.empty(size, dtype=np.int64)
    dense_index_conversion[(1,)](output, num_cpu_threads=1)
    expected = (np.arange(size, dtype=np.int64) % 64) * 4294967291 + 17
    np.testing.assert_array_equal(output, expected)


def test_loop_invariant_math_preserves_frontend_value_protocol():
    from semantic_programs import loop_invariant_logical_math
    value = np.linspace(-2, 2, 9, dtype=np.float32)
    output = np.empty_like(value)
    loop_invariant_logical_math[(1,)](value, output, num_cpu_threads=1)
    np.testing.assert_allclose(output, 4 * np.exp(value), rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("control", ["for", "while"])
def test_canonical_loop_keeps_previous_local_value(tmp_path, control):
    import triton
    from triton.backends.weft.driver import WeftDriver
    value = '!weft_kernel.value<f32, [512], [1]>'
    view = '!weft_kernel.view<!weft_kernel.encoding<"f32", "dense", "dense.f32", []>, [512], [1]>'
    region = '!weft_kernel.slice<!weft_kernel.encoding<"f32", "dense", "dense.f32", []>, [512], [1]>'
    update = f'''%next = "weft_kernel.update"(%state, %two, %i)
        <{{selectors = ["index"]}}> : ({value}, f32, index) -> {value}'''
    if control == "for":
        loop = f'''%changed = scf.for %i = %zero to %end step %one
          iter_args(%state = %original) -> ({value}) {{
          {update}
          scf.yield %next : {value}
        }}'''
    else:
        loop = f'''%position, %changed = scf.while (%i = %zero, %state = %original)
          : (index, {value}) -> (index, {value}) {{
          %active = "weft_kernel.compare"(%i, %end) <{{predicate = "lt"}}> : (index, index) -> i1
          scf.condition(%active) %i, %state : index, {value}
        }} do {{
        ^bb0(%i: index, %state: {value}):
          {update}
          %advance = "weft_kernel.binary"(%i, %one) <{{kind = "add"}}> : (index, index) -> index
          scf.yield %advance, %next : index, {value}
        }}'''
    source = tmp_path / "local_versions.weftir"
    source.write_text(f'''module {{
      "weft_kernel.kernel"() <{{arg_access = ["read", "write", "write"],
        arg_alias_sets = array<i64: 0, 0, 0>, arg_names = ["X", "Y", "Old"],
        shape_symbols = [], strided_args = [], source = "canonical regression", sym_name = "local_versions"}}> ({{
      ^bb0(%x: {view}, %y: {view}, %old: {view}):
        %root = "weft_kernel.root_domain"() : () -> !weft_kernel.domain<"root", 0, -1, 0, "root", "exact">
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %end = arith.constant 512 : index
        %two = arith.constant 2.0 : f32
        %xs = "weft_kernel.slice"(%x) <{{selectors = ["all"]}}> : ({view}) -> {region}
        %input = "weft_kernel.admit"(%xs) : ({region}) -> {value}
        %original = "weft_kernel.new"(%input) <{{initialized = true}}> : ({value}) -> {value}
        {loop}
        %ys = "weft_kernel.slice"(%y) <{{selectors = ["all"]}}> : ({view}) -> {region}
        %saved = "weft_kernel.slice"(%old) <{{selectors = ["all"]}}> : ({view}) -> {region}
        "weft_kernel.commit"(%changed, %ys) : ({value}, {region}) -> ()
        "weft_kernel.commit"(%original, %saved) : ({value}, {region}) -> ()
        "weft_kernel.return"() : () -> ()
      }}) : () -> ()
    }}''')
    kernel = triton.compile(str(source), target=WeftDriver().get_current_target(),
                            options={"num_cpu_threads": 1})
    original = np.arange(512, dtype=np.float32)
    result, saved = np.empty_like(original), np.empty_like(original)
    kernel[(1,)](original, result, saved)
    np.testing.assert_array_equal(result, np.full_like(original, 2))
    np.testing.assert_array_equal(saved, original)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("ordering", ["relaxed", "acquire", "release", "acq_rel", "seq_cst"])
def test_canonical_scalar_atomic_returns_previous_value(tmp_path, dtype, ordering):
    import triton
    from triton.backends.weft.driver import WeftDriver
    bits = np.dtype(dtype).itemsize * 8
    signed = np.issubdtype(dtype, np.signedinteger)
    family = ("i" if signed else "u") + str(bits)
    element = ("si" if signed else "ui") + str(bits)
    encoding = f'!weft_kernel.encoding<"{family}", "dense", "dense.{family}", []>'
    view = f'!weft_kernel.view<{encoding}, [1], [1]>'
    region = f'!weft_kernel.slice<{encoding}, [], []>'
    source = tmp_path / "atomic.weftir"
    source.write_text(f'''module {{
      "weft_kernel.kernel"() <{{arg_access = ["readwrite", "write"],
        arg_alias_sets = array<i64: 0, 0>, arg_names = ["X", "Y"],
        shape_symbols = [], strided_args = [], source = "canonical regression", sym_name = "scalar_atomic"}}> ({{
      ^bb0(%x: {view}, %y: {view}):
        %root = "weft_kernel.root_domain"() : () -> !weft_kernel.domain<"root", 0, -1, 0, "root", "exact">
        %index = arith.constant 0 : index
        %value = "weft_kernel.constant"() <{{value = 7 : {element}}}> : () -> {element}
        %xs = "weft_kernel.slice"(%x, %index) <{{selectors = ["index"]}}> : ({view}, index) -> {region}
        %ys = "weft_kernel.slice"(%y, %index) <{{selectors = ["index"]}}> : ({view}, index) -> {region}
        %old = "weft_kernel.atomic_rmw"(%value, %xs) <{{kind = "add", ordering = "{ordering}"}}> : ({element}, {region}) -> {element}
        "weft_kernel.commit"(%old, %ys) : ({element}, {region}) -> ()
        "weft_kernel.return"() : () -> ()
      }}) : () -> ()
    }}''')
    kernel = triton.compile(str(source), target=WeftDriver().get_current_target(),
                            options={"num_cpu_threads": 1})
    value = np.array([np.iinfo(dtype).max - 3], dtype=dtype)
    original = value.copy()
    output = np.empty_like(value)
    kernel[(1,)](value, output)
    np.testing.assert_array_equal(output, original)
    np.testing.assert_array_equal(value, original + np.array([7], dtype=dtype))
    second, second_output = np.array([19], dtype=dtype), np.empty_like(value)
    kernel[(1,)](X=second, Y=second_output)
    np.testing.assert_array_equal(second, [26])
    np.testing.assert_array_equal(second_output, [19])


@pytest.mark.parametrize("format", ["f8E4M3FN", "f8E5M2"])
def test_canonical_fp8_widening_all_encodings(tmp_path, format):
    import triton
    from triton.backends.weft.driver import WeftDriver
    source = tmp_path / "fp8.weftir"
    source.write_text(f'''module {{
      "weft_kernel.kernel"() <{{arg_access = ["read", "write"],
        arg_alias_sets = array<i64: 0, 0>, arg_names = ["X", "Y"],
        shape_symbols = [], strided_args = [], source = "canonical regression", sym_name = "fp8_widen"}}> ({{
      ^bb0(%x: !weft_kernel.view<!weft_kernel.encoding<"u8", "dense", "dense.u8", []>, [256], [1]>,
           %y: !weft_kernel.view<!weft_kernel.encoding<"f32", "dense", "dense.f32", []>, [256], [1]>):
        %root = "weft_kernel.root_domain"() : () -> !weft_kernel.domain<"root", 0, -1, 0, "root", "exact">
        %bits = "weft_kernel.admit"(%x) : (!weft_kernel.view<!weft_kernel.encoding<"u8", "dense", "dense.u8", []>, [256], [1]>) -> !weft_kernel.value<ui8, [256], [1]>
        %small = "weft_kernel.bitcast"(%bits) : (!weft_kernel.value<ui8, [256], [1]>) -> !weft_kernel.value<{format}, [256], [1]>
        %wide = "weft_kernel.cast"(%small) : (!weft_kernel.value<{format}, [256], [1]>) -> !weft_kernel.value<f32, [256], [1]>
        "weft_kernel.commit"(%wide, %y) : (!weft_kernel.value<f32, [256], [1]>, !weft_kernel.view<!weft_kernel.encoding<"f32", "dense", "dense.f32", []>, [256], [1]>) -> ()
        "weft_kernel.return"() : () -> ()
      }}) : () -> ()
    }}''')
    kernel = triton.compile(str(source), target=WeftDriver().get_current_target(),
                            options={"num_cpu_threads": 1})
    bits = np.arange(256, dtype=np.uint8)
    output = np.empty(256, dtype=np.float32)
    kernel[(1,)](bits, output)
    mantissa, bias = (3, 7) if format == "f8E4M3FN" else (2, 15)
    expected = []
    for byte in range(256):
        exponent = (byte & 127) >> mantissa
        fraction = byte & ((1 << mantissa) - 1)
        value = np.ldexp(fraction / (1 << mantissa) + bool(exponent),
                         max(1, exponent) - bias)
        if format == "f8E4M3FN" and (byte & 127) == 127:
            value = np.nan
        if format == "f8E5M2" and exponent == 31:
            value = np.inf if fraction == 0 else np.nan
        expected.append(-value if byte & 128 else value)
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(np.signbit(output[~np.isnan(expected)]),
                                  np.signbit(expected[~np.isnan(expected)]))


def test_canonical_bfloat_arithmetic_preserves_rounding(tmp_path):
    import triton
    from triton.backends.weft.driver import WeftDriver
    source = tmp_path / "bfloat.weftir"
    source.write_text('''module {
      "weft_kernel.kernel"() <{arg_access = ["read", "read", "write"],
        arg_alias_sets = array<i64: 0, 0, 0>, arg_names = ["X", "Y", "Z"],
        shape_symbols = [], strided_args = [], source = "canonical regression", sym_name = "bfloat_product"}> ({
      ^bb0(%x: !weft_kernel.view<!weft_kernel.encoding<"u16", "dense", "dense.u16", []>, [32], [1]>,
           %y: !weft_kernel.view<!weft_kernel.encoding<"u16", "dense", "dense.u16", []>, [32], [1]>,
           %z: !weft_kernel.view<!weft_kernel.encoding<"u16", "dense", "dense.u16", []>, [32], [1]>):
        %root = "weft_kernel.root_domain"() : () -> !weft_kernel.domain<"root", 0, -1, 0, "root", "exact">
        %a = "weft_kernel.admit"(%x) : (!weft_kernel.view<!weft_kernel.encoding<"u16", "dense", "dense.u16", []>, [32], [1]>) -> !weft_kernel.value<ui16, [32], [1]>
        %b = "weft_kernel.admit"(%y) : (!weft_kernel.view<!weft_kernel.encoding<"u16", "dense", "dense.u16", []>, [32], [1]>) -> !weft_kernel.value<ui16, [32], [1]>
        %af = "weft_kernel.bitcast"(%a) : (!weft_kernel.value<ui16, [32], [1]>) -> !weft_kernel.value<bf16, [32], [1]>
        %bf = "weft_kernel.bitcast"(%b) : (!weft_kernel.value<ui16, [32], [1]>) -> !weft_kernel.value<bf16, [32], [1]>
        %product = "weft_kernel.binary"(%af, %bf) <{kind = "mul"}> : (!weft_kernel.value<bf16, [32], [1]>, !weft_kernel.value<bf16, [32], [1]>) -> !weft_kernel.value<bf16, [32], [1]>
        %result = "weft_kernel.bitcast"(%product) : (!weft_kernel.value<bf16, [32], [1]>) -> !weft_kernel.value<ui16, [32], [1]>
        "weft_kernel.commit"(%result, %z) : (!weft_kernel.value<ui16, [32], [1]>, !weft_kernel.view<!weft_kernel.encoding<"u16", "dense", "dense.u16", []>, [32], [1]>) -> ()
        "weft_kernel.return"() : () -> ()
      }) : () -> ()
    }''')
    kernel = triton.compile(str(source), target=WeftDriver().get_current_target(),
                            options={"num_cpu_threads": 1})
    left = np.array([0, 1, 0x007f, 0x0080, 0x3f80, 0x3f81, 0x3fff, 0x4001,
                     0x7f7f, 0x7f80, 0x7fc1, 0x8000, 0x8001, 0xbf81, 0xff80, 0xffc1] * 2, dtype=np.uint16)
    right = np.array([0x3f80, 0x3f00, 0x3f81, 0x4000] * 8, dtype=np.uint16)
    output = np.empty_like(left)
    kernel[(1,)](left, right, output)
    with np.errstate(over="ignore", invalid="ignore"):
        product = (left.astype(np.uint32) << 16).view(np.float32) * (right.astype(np.uint32) << 16).view(np.float32)
    raw = product.view(np.uint32)
    expected = ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
    finite = ~np.isnan(product)
    np.testing.assert_array_equal(output[finite], expected[finite])
    assert np.isnan((output[~finite].astype(np.uint32) << 16).view(np.float32)).all()


@pytest.mark.parametrize("row", [0, 2])
@pytest.mark.parametrize("offset", [0, 8])
def test_canonical_field_store_through_record_subview(tmp_path, row, offset):
    import triton
    import triton.language as tl
    from triton.backends.weft.driver import WeftDriver
    from triton.integrations import ggml_formats as formats

    encoded = '!weft_kernel.encoding<"Q8_0", "base", "Q8_0", []>'
    dense = '!weft_kernel.encoding<"i8", "dense", "dense.i8", []>'
    view = f'!weft_kernel.view<{encoded}, [3, 2, 32], [1, 2, 3]>'
    selected = f'!weft_kernel.slice<{encoded}, [1, 2, 32], [1, 2, 3]>'
    record = f'!weft_kernel.slice<{encoded}, [32], [3]>'
    field = f'!weft_kernel.slice<{dense}, [32], [3]>'
    piece = f'!weft_kernel.slice<{dense}, [8], [3]>'
    value = '!weft_kernel.value<si8, [8], [3]>'
    source = tmp_path / 'record_subview.weftir'
    source.write_text(f'''module {{
      "weft_kernel.encoding_decl"() <{{alignment = 2 : i64, bit_order = "lsb_first", byte_order = "little",
        elements = 32 : i64, field_bit_offsets = array<i64: 0, 16>, field_layouts = [[{{kind = "natural"}}], [{{kind = "natural"}}]],
        field_names = ["d", "q"], field_shapes = [array<i64>, array<i64: 32>], field_storage_bits = array<i64: 16, 256>,
        field_types = [f16, si8], kind = "base", layout_identity = "Q8_0", padding = array<i64>,
        storage_bits = 272 : i64, sym_name = "Q8_0"}}> : () -> ()
      "weft_kernel.kernel"() <{{arg_access = ["read", "readwrite"], arg_alias_sets = array<i64: 0, 0>,
        arg_names = ["X", "Y"], shape_symbols = [], strided_args = [], source = "canonical regression", sym_name = "record_subview"}}> ({{
      ^bb0(%x: !weft_kernel.view<{dense}, [8], [3]>, %y: {view}):
        %root = "weft_kernel.root_domain"() : () -> !weft_kernel.domain<"root", 0, -1, 0, "root", "exact">
        %row = arith.constant {row} : index
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %window = "weft_kernel.subview"(%y, %row) <{{offsets = array<i64: -1, 0, 0>, extents = array<i64: 1, 2, 32>}}> : ({view}, index) -> {selected}
        %record = "weft_kernel.slice"(%window, %zero, %one) <{{selectors = ["index", "index", "all"]}}> : ({selected}, index, index) -> {record}
        %field = "weft_kernel.field"(%record) <{{name = "q"}}> : ({record}) -> {field}
        %part = "weft_kernel.subview"(%field) <{{offsets = array<i64: {offset}>, extents = array<i64: 8>}}> : ({field}) -> {piece}
        %input = "weft_kernel.admit"(%x) : (!weft_kernel.view<{dense}, [8], [3]>) -> {value}
        "weft_kernel.commit"(%input, %part) : ({value}, {piece}) -> ()
        "weft_kernel.return"() : () -> ()
      }}) : () -> ()
    }}''')
    kernel = triton.compile(str(source), target=WeftDriver().get_current_target(),
                            options={"num_cpu_threads": 1})
    source_bytes = np.arange(-8, 0, dtype=np.int8)
    storage = np.full((3, 2, 34), 77, dtype=np.uint8)
    expected = storage.copy()
    expected[row, 1, 2 + offset:10 + offset] = source_bytes.view(np.uint8)
    kernel[(1,)](source_bytes, tl.View(storage, encoding=formats.Q8_0, shape=(3, 2, 32)))
    np.testing.assert_array_equal(storage, expected)
