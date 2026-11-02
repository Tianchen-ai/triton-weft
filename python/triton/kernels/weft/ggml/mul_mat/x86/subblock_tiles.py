from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from ...quantize.x86.blocks32 import quantize


@triton.jit
def project(W, X, Xq, Y, M, N, K, BN: tl.constexpr, BITS: tl.constexpr,
            codebook=None, scale=None):
    """Reuse a decoded column tile across activation rows within the call."""
    for row in range(M):
        quantize(X[row], Xq[row], K, False, 4)
    RECORD: tl.constexpr = 128 if BITS else 64
    GROUP: tl.constexpr = 32 if BITS else 16
    lane = tl.arange(0, GROUP, dtype=tl.u32, axis="k")
    with tl.level.cols(N, group=BN) as columns:
        for row in range(M):
            zero = tl.state(tl.f32, [BN], init=0)
            tl.store(Y[row, columns], zero)
        with tl.level.blocks(K, extent=RECORD) as block:
            if tl.extent(W[columns, block], axis=1) >= RECORD:
                w = tl.load(W[columns, block])
                for part in tl.static_range(RECORD // GROUP):
                    q = w.q[:, lane + tl.u32(part * GROUP)]
                    if BITS:
                        weight = tl.reshape(q, (BN, GROUP), axes=("N", "k"), order=("N", "k"))
                        wd = tl.f32(w.d)
                    else:
                        decoded = tl.lookup(codebook, tl.u8(q), bounds="in_bounds")
                        weight = tl.reshape(decoded, (BN, GROUP), axes=("N", "k"), order=("N", "k"))
                        wd = tl.lookup(scale, tl.u32(w.d[:, part]), bounds="in_bounds")
                    for row in range(M):
                        x = tl.load(tl.subview(Xq[row, block], offsets=(part * GROUP // 32 * 32,), extents=(32,)))
                        activation = x.q[lane + tl.u32(part * GROUP % 32)]
                        value = tl.reduce_dot(weight, activation, over="k", acc_dtype=tl.i32)
                        if BITS:
                            value = value * tl.i32(2) - tl.sum(tl.i32(activation), 0)
                        prior = tl.load(Y[row, columns])
                        tl.store(Y[row, columns], prior + (wd * tl.f32(x.d)) * tl.f32(value))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q1_0(W: tl.View[ggml_formats.Q1_0, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)], BN: tl.constexpr = 8):
    project(W, X, Xq, Y, M, N, K, BN, True)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def nvfp4(W: tl.View[ggml_formats.NVFP4, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
           scale: tl.View[tl.f32, (256,)], Y: tl.View[tl.f32, (M, N)], BN: tl.constexpr = 8):
    table = tl.load(codebook)
    project(W, X, Xq, Y, M, N, K, BN, False, table, scale)
