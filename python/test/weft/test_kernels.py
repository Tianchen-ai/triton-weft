"""Finite full-output checks of the installed kernel corpus and matrix paths."""
import numpy as np
import pytest
import platform
import triton.language as tl


@pytest.mark.parametrize("example", ["projection", "welford", "attention", "gemm"])
def test_complete_examples(example):
    from importlib import import_module
    import_module("examples.weft.kernels." + example).main()


@pytest.mark.parametrize("shape", [(1, 1, 1), (15, 17, 31), (16, 16, 32), (17, 15, 33), (31, 33, 63), (32, 32, 64), (33, 31, 65)])
@pytest.mark.parametrize("dtype", ["i8", "bf16"])
@pytest.mark.skipif(platform.machine() != "x86_64", reason="The tested matrix implementation uses Intel AMX")
def test_matrix_complete_output(shape, dtype):
    from triton.kernels.weft.dense.x86 import integer_matmul, bfloat_matmul
    rows, columns, width = shape
    rng = np.random.default_rng(19)
    if dtype == "i8":
        left = rng.integers(-128, 128, (rows, width), dtype=np.int8)
        right = rng.integers(-128, 128, (columns, width), dtype=np.int8)
        output = np.empty((rows, columns), np.int32)
        expected = left.astype(np.int64) @ right.astype(np.int64).T
        compiled = integer_matmul.full_reduction[(1,)](left, right, output, M=rows, N=columns, K=width)
        np.testing.assert_array_equal(output, expected.astype(np.int32))
    else:
        left = rng.integers(-32, 33, (rows, width)).astype(np.float32) / 4
        right = rng.integers(-32, 33, (columns, width)).astype(np.float32) / 4
        left_bits = (left.view(np.uint32) >> 16).astype(np.uint16)
        right_bits = (right.view(np.uint32) >> 16).astype(np.uint16)
        output = np.empty((rows, columns), np.float32)
        expected = left.astype(np.float64) @ right.astype(np.float64).T
        compiled = bfloat_matmul.full_reduction[(1,)](
            tl.View(left_bits, encoding=tl.bf16, shape=left.shape),
            tl.View(right_bits, encoding=tl.bf16, shape=right.shape), output,
            M=rows, N=columns, K=width)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    assert "weft_kernel" in compiled.asm["weftir"]


@pytest.mark.skipif(platform.machine() != "x86_64", reason="requires the x86 oneDNN BRGEMM backend")
def test_brgemm_packs_adjacent_signed_bytes_in_k_order():
    from triton import cpu
    from triton.kernels.weft.dense.x86 import integer_matmul
    from weft_core.compiler import x86_target_capabilities

    target = cpu.target()
    caps = x86_target_capabilities(target.march, target.abi, target.vlen_bits)
    if not caps["onednn_brgemm"]["s8"]:
        pytest.skip("oneDNN signed-byte BRGEMM is unavailable")
    m, n, k = 32, 64, 128
    rng = np.random.default_rng(103)
    left = rng.integers(-128, 128, (m, k), dtype=np.int8)
    right = rng.integers(-128, 128, (n, k), dtype=np.int8)
    output = np.empty((m, n), dtype=np.int32)
    kernel = integer_matmul.full_reduction[(1,)](
        left, right, output, M=m, N=n, K=k, BM=m, BN=n,
        contraction_backend="onednn-brgemm", num_cpu_threads=1)
    assert "__weft_onednn_brgemm_execute" in kernel.asm["llir"]
    np.testing.assert_array_equal(output, left.astype(np.int32) @ right.astype(np.int32).T)


@pytest.mark.skipif(platform.machine() != "x86_64", reason="requires the x86 oneDNN BRGEMM backend")
def test_brgemm_instance_keeps_float_argument_on_first_call(tmp_path):
    import ctypes
    import shutil
    from triton import cpu
    from triton.kernels.weft.dense.contraction import dense_projection_bias
    from weft_core.compiler import x86_target_capabilities

    target = cpu.target()
    if not x86_target_capabilities(target.march, target.abi, target.vlen_bits)["onednn_brgemm"]["f32"]:
        pytest.skip("oneDNN FP32 BRGEMM is unavailable")
    size = 32
    x = np.ones((size, size), dtype=np.float32)
    bias = np.ones(size, dtype=np.float32)
    output = np.empty_like(x)
    scale = 0.5
    kernel = dense_projection_bias[(1,)](x, x, bias, output, scale, contraction_backend="onednn-brgemm",
                                         num_cpu_threads=1)
    assert "__weft_onednn_brgemm_execute" in kernel.asm["llir"]
    directory = tmp_path / "projection"
    kernel.export(directory)
    arrays = (x, x, bias, output)
    strides = [stride // a.itemsize for a in arrays for stride in a.strides]
    abi = kernel.metadata.single_instance_abi
    extents = [size] * len(abi["shape_parameters"])
    geometry = extents + strides
    # Each separately loaded image resolves its own dynamic TLS on the first call.
    images = []
    for i in range(32):
        image = directory / f"projection_{i}.so"
        shutil.copyfile(directory / "kernel.so", image)
        library = ctypes.CDLL(str(image))
        images.append(library)
        instance = getattr(library, abi["symbol"])
        instance.argtypes = [ctypes.c_void_p] * len(arrays) + [ctypes.c_float] + [ctypes.c_size_t] * len(geometry)
        instance.restype = None
        output.fill(np.nan)
        instance(*(a.ctypes.data for a in arrays), scale, *geometry)
        np.testing.assert_array_equal(output, size * scale + 1.0)


@pytest.mark.skipif(platform.machine() != "x86_64", reason="exercises x86 contraction operands")
def test_reshaped_contraction_snapshot_reads_strided_inputs():
    from triton.kernels.weft.dense.contraction import renamed_projection

    rng = np.random.default_rng(29)
    left = rng.integers(-8, 9, (32, 258)).astype(np.float32)[:, 1::2]
    right = rng.integers(-8, 9, (65, 129)).astype(np.float32)[::2]
    output = np.full((32, 33), np.nan, dtype=np.float32)
    renamed_projection[(1,)](left, right, output, num_cpu_threads=1)
    np.testing.assert_array_equal(output, left @ right.T)


@pytest.mark.parametrize("dtype", ["i8", "bf16"])
@pytest.mark.skipif(platform.machine() != "x86_64", reason="The tested matrix implementation uses Intel AMX")
def test_aligned_matrix_complete_output(dtype):
    from triton import cpu
    extension = "amx-int8" if dtype == "i8" else "amx-bf16"
    if extension not in cpu.target().march.split("+"):
        pytest.skip(f"CPU lacks {extension}")
    from triton.kernels.weft.dense.x86 import integer_matmul, bfloat_matmul
    rows, columns, width = 80, 80, 64
    rng = np.random.default_rng(19)
    if dtype == "i8":
        left = rng.integers(-128, 128, (rows, width), dtype=np.int8)
        right = rng.integers(-128, 128, (columns, width), dtype=np.int8)
        output = np.empty((rows, columns), np.int32)
        compiled = integer_matmul.full_reduction[(1,)](left, right, output,
            M=rows, N=columns, K=width, BM=80, BN=80, ENGINE="matrix")
        np.testing.assert_array_equal(output, (left.astype(np.int64) @ right.astype(np.int64).T).astype(np.int32))
    else:
        left = rng.integers(-32, 33, (rows, width)).astype(np.float32) / 4
        right = rng.integers(-32, 33, (columns, width)).astype(np.float32) / 4
        output = np.empty((rows, columns), np.float32)
        compiled = bfloat_matmul.full_reduction[(1,)](
            tl.View((left.view(np.uint32) >> 16).astype(np.uint16), encoding=tl.bf16, shape=left.shape),
            tl.View((right.view(np.uint32) >> 16).astype(np.uint16), encoding=tl.bf16, shape=right.shape), output,
            M=rows, N=columns, K=width, BM=80, BN=80)
        np.testing.assert_allclose(output, left.astype(np.float64) @ right.astype(np.float64).T,
                                   rtol=1e-5, atol=1e-5)
    assert "llvm.x86.tdp" in compiled.asm["llir"]
