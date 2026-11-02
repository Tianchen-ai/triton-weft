"""Online attention over key blocks, checked against the complete dense task."""
import numpy as np
from triton.kernels.weft.attention.online_f32 import online_blocked


def main():
    rng = np.random.default_rng(19)
    query = rng.normal(size=(3, 16)).astype(np.float32)
    keys = rng.normal(size=(65, 16)).astype(np.float32)
    values = rng.normal(size=(65, 16)).astype(np.float32)
    output = np.empty_like(query)
    scale = np.float32(0.25)
    compiled = online_blocked[(1,)](query, keys, values, output, scale,
                                    BQ=1, BK=32, D=16, num_cpu_threads=1)
    score = query.astype(np.float64) @ keys.astype(np.float64).T * scale
    probability = np.exp(score - score.max(axis=1)[:, None])
    reference = probability @ values / probability.sum(axis=1)[:, None]
    np.testing.assert_allclose(output, reference, rtol=2e-4, atol=2e-5)
    print("Online attention: complete output matches dense softmax attention")
    return compiled


if __name__ == "__main__":
    main()
