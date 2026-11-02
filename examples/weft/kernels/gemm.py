"""A complete tiled GEMM with output and reduction tails."""
import numpy as np
from triton.kernels.weft.dense.basic.gemm_f32 import gemm_f32


def main():
    rng = np.random.default_rng(19)
    left = rng.normal(size=(17, 65)).astype(np.float32)
    right = rng.normal(size=(19, 65)).astype(np.float32)
    output = np.empty((17, 19), dtype=np.float32)
    compiled = gemm_f32[(1,)](left, right, output, MC=16, NC=16, MR=4, NR=4, num_cpu_threads=1)
    np.testing.assert_allclose(output, left.astype(np.float64) @ right.astype(np.float64).T,
                               rtol=2e-4, atol=2e-5)
    print("GEMM: complete output matches the reference", output.shape)
    return compiled


if __name__ == "__main__":
    main()
