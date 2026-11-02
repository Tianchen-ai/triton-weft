from __future__ import annotations

import triton
import triton.language as tl

inf = tl.constexpr(float("inf"))


@triton.jit(model="weft")
def top_k_f32(X: tl.View[tl.f32, (N,)], out: tl.View[tl.i32, (K_,)]):
    heap = tl.state(tl.f32, [K_], init=-inf)
    idx = tl.state(tl.i32, [K_], init=-1)
    for i in range(N):
        score = X[i]
        if score > heap[K_ - 1]:
            pos = K_ - 1
            j = K_ - 1
            while j > 0:
                if score > heap[j - 1]:
                    pos = j - 1
                j -= 1
            for j in range(K_ - 1, pos, -1):
                heap[j] = heap[j - 1]
                idx[j] = idx[j - 1]
            heap[pos] = score
            idx[pos] = tl.i32(i)
    tl.store(out, idx)
