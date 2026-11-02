from __future__ import annotations

from triton.integrations import ggml_formats
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_q4_k_persistent(
    W: tl.View[ggml_formats.Q4K_I[16], (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        with tl.level.rows(N, group=16) as nb:
            f32_acc = tl.state(tl.f32, [16], init=0)
            with tl.level.blocks(K, extent=256) as kb:
                w = tl.load(W[nb, kb])
                x = tl.load(Xq[row, kb])
                i32_acc = tl.state(tl.i32, [16], init=0)
                with tl.level.subtiles(extent=32) as s:
                    p16 = tl.mac_groups(w.q[s], x.q[s], n=4, into=tl.i16)
                    i32_acc += tl.reduce(tl.widen(p16, tl.i32)) * w.sc[s]
                mins = tl.sum_pairs(x.bsum)
                min_term = tl.reduce_dot(w.m, mins, over="k")
                f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
            tl.store(Y[row, nb], f32_acc)
