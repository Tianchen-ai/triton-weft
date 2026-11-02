"""Block summaries, coupled updates and a double-precision full-output check."""
import numpy as np
from triton.kernels.weft.programs.statistics import welford_blocked


def main():
    value = np.random.default_rng(19).normal(size=(3, 257)).astype(np.float32)
    output = np.empty((3, 3), dtype=np.float32)
    compiled = welford_blocked[(1,)](value, output, ROWS=1, BLOCK=64, num_cpu_threads=1)
    reference = value.astype(np.float64)
    np.testing.assert_array_equal(output[:, 0], 257)
    np.testing.assert_allclose(output[:, 1], reference.mean(axis=1), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(output[:, 2], ((reference - reference.mean(axis=1)[:, None]) ** 2).sum(axis=1),
                               rtol=1e-4, atol=1e-5)
    print("Welford: count, mean and M2 match the complete reference")
    return compiled


if __name__ == "__main__":
    main()
