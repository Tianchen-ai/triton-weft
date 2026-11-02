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


@pytest.mark.parametrize("dtype", ["i8", "bf16"])
@pytest.mark.skipif(platform.machine() != "x86_64", reason="The tested matrix implementation uses Intel AMX")
def test_aligned_matrix_complete_output(dtype):
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
