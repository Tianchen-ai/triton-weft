"""Quantized projection with normal input preparation and a complete reference."""
import numpy as np
import triton.language as tl
from triton.integrations.ggml_formats import Q4_K, Q8_K
from triton.kernels.weft.ggml.blocks.mul_mat.organized_projection import q4_projection


def main():
    rows, outputs, width = 3, 5, 512
    generator = np.random.default_rng(19)
    activation = generator.integers(-127, 128, (rows, width)).astype(np.float32)
    activation[:, 0] = 127
    activation[:, 256] = 127
    weights = np.zeros((outputs, width // 256, 144), dtype=np.uint8)
    factors = np.arange(1, outputs + 1, dtype=np.float16) / np.float16(16)
    for column in range(outputs):
        for block in range(width // 256):
            weights[column, block, :2] = np.array([factors[column]], np.float16).view(np.uint8)
            weights[column, block, 4:8] = 1
            weights[column, block, 12:16] = 1
            weights[column, block, 16:] = 0x21
    workspace = np.empty((rows, width // 256, 292), dtype=np.uint8)
    output = np.empty((rows, outputs), dtype=np.float32)
    compiled = q4_projection[(1,)](
        tl.View(weights, encoding=Q4_K, shape=(outputs, width)), activation,
        tl.View(workspace, encoding=Q8_K, shape=(rows, width)), output,
        MR=1, NR=4, MC=1, SHARED=True, GROUPED=True, RETAIN_OUTPUT=True,
        num_cpu_threads=1)
    codes = np.where(np.arange(width) % 64 < 32, 1, 2).astype(np.float32)
    reference = activation @ (factors.astype(np.float32)[:, None] * codes).T
    np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-4)
    print("Q4_K projection: complete output matches the reference", output.shape)
    return compiled


if __name__ == "__main__":
    main()
