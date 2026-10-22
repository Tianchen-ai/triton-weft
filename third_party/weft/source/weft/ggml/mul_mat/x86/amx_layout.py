from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as tl
from weft.ggml.quantize.x86.blocks32 import quantize, BlockSum32


@triton.encoding
class AMXQ4_0:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    q: tl.u4[32] @ tl.grouped(elements=8) @ tl.bit_layers(elements=4, order=tl.lo_first)
    d: tl.f16


@triton.derive
def AMXQ4_0Rows(W: tl.View[AMXQ4_0, (N, K)], *, rows: tl.static[int]) -> tl.View[AMXQ4_0Rows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"q": 4, "d": 2})


@triton.encoding
class BlockScale32:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 4
    elements = 32
    value: tl.f32


@triton.derive
def BlockScale32Rows(S: tl.View[BlockScale32, (M, K)], *, rows: tl.static[int]) -> tl.View[BlockScale32Rows[rows], (M, K)]:
    return tl.interleave(S, rows=rows, field_chunks={"value": 4})


@triton.jit
def compute(W, Xq, Y, N, K, BN: tl.constexpr, Sums=None):
    with tl.level.cols(N, group=BN) as columns:
        partial = tl.state(tl.f32, [BN], init=0)
        with tl.level.blocks(K, extent=32) as block:
            w, x = W[columns, block], Xq[block]
            weight, activation = tl.load(w.q), tl.load(x.q)
            products = tl.reduce_dot(weight, activation, over="K", acc_dtype=tl.i32)
            if Sums is None:
                activation_sum = tl.sum(tl.i32(activation), 0)
            else:
                activation_sum = tl.i32(tl.load(Sums[block].value))
            products -= tl.i32(8) * activation_sum
            partial += tl.f32(products) * (tl.f32(tl.load(w.d)) * tl.f32(tl.load(x.d)))
        tl.store(Y[columns], partial)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_0_decode(W: tl.View[AMXQ4_0Rows[16], (N, K)], Xq: tl.View[ggml.Q8_0, (K,)],
                Y: tl.View[tl.f32, (N,)], BN: tl.constexpr = 32, PARTS: tl.constexpr = 1):
    """Consume the CPU backend's existing packed weights and quantized row."""
    tl.static_assert(PARTS == 1 or PARTS == 2)
    if PARTS == 1:
        compute(W, Xq, Y, N, K, BN)
    else:
        compute_pairs(W, Xq, Y, N, K, BN)


@triton.jit
def activation_sums(Xq, Sums, K):
    with tl.level.blocks(K, extent=32) as block:
        value = tl.sum(tl.i32(tl.load(Xq[block].q)), 0)
        tl.store(Sums[block].value, tl.i16(value))


@triton.jit(model="weft", alias_groups={"Y": "output", "Sums": "scratch"})
def q4_0_decode_reuse(W: tl.View[AMXQ4_0Rows[16], (N, K)], Xq: tl.View[ggml.Q8_0, (K,)],
                      Y: tl.View[tl.f32, (N,)], Sums: tl.View[BlockSum32, (K,)], BN: tl.constexpr = 64):
    activation_sums(Xq, Sums, K)
    compute(W, Xq, Y, N, K, BN, Sums)


@triton.jit(model="weft", alias_groups={"Y": "output", "Sums": "scratch"})
def q4_0_decode_tile_reuse(W: tl.View[AMXQ4_0Rows[16], (N, K)], Xq: tl.View[ggml.Q8_0, (M, K)],
                           Y: tl.View[tl.f32, (M, N)], Sums: tl.View[BlockSum32, (K,)], BN: tl.constexpr = 64):
    for row in range(M):
        activation_sums(Xq[row], Sums, K)
        compute(W, Xq[row], Y[row], N, K, BN, Sums)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_0_decode_tile(W: tl.View[AMXQ4_0Rows[16], (N, K)], Xq: tl.View[ggml.Q8_0, (M, K)],
                     Y: tl.View[tl.f32, (M, N)], BN: tl.constexpr = 64, PARTS: tl.constexpr = 1):
    tl.static_assert(PARTS == 1 or PARTS == 2)
    for row in range(M):
        if PARTS == 1:
            compute(W, Xq[row], Y[row], N, K, BN)
        else:
            compute_pairs(W, Xq[row], Y[row], N, K, BN)


@triton.jit
def compute_pairs(W, Xq, Y, N, K, BN: tl.constexpr):
    with tl.level.cols(N, group=BN) as columns:
        first = tl.state(tl.f32, [BN], init=0)
        second = tl.state(tl.f32, [BN], init=0)
        with tl.level.blocks(K, extent=64) as block:
            for part in tl.static_range(2):
                if part * 32 < tl.extent(Xq[block], axis=0):
                    w = tl.subview(W[columns, block], offsets=(0, part * 32), extents=(BN, 32))
                    x = tl.subview(Xq[block], offsets=(part * 32,), extents=(32,))
                    weight, activation = tl.load(w.q), tl.load(x.q)
                    products = tl.reduce_dot(weight, activation, over="K", acc_dtype=tl.i32)
                    products -= tl.i32(8) * tl.sum(tl.i32(activation), 0)
                    scaled = tl.f32(products) * (tl.f32(tl.load(w.d)) * tl.f32(tl.load(x.d)))
                    if part == 0:
                        first += scaled
                    else:
                        second += scaled
        tl.store(Y[columns], first + second)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_0_projection(W: tl.View[AMXQ4_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                    Xq: tl.View[ggml.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)],
                    BN: tl.constexpr = 32):
    for row in range(M):
        quantize(X[row], Xq[row], K, False, 4)
        compute(W, Xq[row], Y[row], N, K, BN)


@triton.jit
def compute_rows(W, Xq, Y, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
    with tl.level.cols(N, group=BN) as columns:
        with tl.level.rows(M, group=BM) as rows:
            partial = tl.state(tl.f32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=32) as block:
                w, x = W[columns, block], Xq[rows, block]
                weight, activation = tl.i8(tl.load(w.q)) - tl.i8(8), tl.load(x.q)
                products = tl.reduce_dot(activation, weight, over="K", acc_dtype=tl.i32)
                scale = tl.f32(tl.load(x.d)) * tl.f32(tl.load(w.d))
                partial += scale * tl.f32(products)
            tl.store(Y[rows, columns], partial)


@triton.jit(model="weft", alias_groups={"Y": "output"}, strided=("Y",))
def q4_0_prefill(W: tl.View[AMXQ4_0Rows[16], (N, K)], Xq: tl.View[ggml.Q8_0, (M, K)],
                 Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 4, BN: tl.constexpr = 16):
    compute_rows(W, Xq, Y, M, N, K, BM, BN)


@triton.jit(model="weft", alias_groups={"Y": "output", "Scales": "workspace"}, strided=("Y",))
def q4_0_prefill_scales(W: tl.View[AMXQ4_0Rows[16], (N, K)], Xq: tl.View[ggml.Q8_0, (M, K)],
                       Y: tl.View[tl.f32, (M, N)], Scales: tl.View[BlockScale32Rows[32], (M, K)],
                       BM: tl.constexpr = 32, BN: tl.constexpr = 32):
    """Share exact widened activation scales across output columns inside the call."""
    tl.static_assert(BM == 32)
    with tl.level.rows(M, group=BM) as rows:
        with tl.level.blocks(K, extent=32) as scale_block:
            tl.store(Scales[rows, scale_block].value, tl.f32(tl.load(Xq[rows, scale_block].d)))
        with tl.level.cols(N, group=BN) as columns:
            partial = tl.state(tl.f32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=32) as block:
                w, x = W[columns, block], Xq[rows, block]
                weight, activation = tl.i8(tl.load(w.q)) - tl.i8(8), tl.load(x.q)
                products = tl.reduce_dot(activation, weight, over="K", acc_dtype=tl.i32)
                scale = tl.load(Scales[rows, block].value) * tl.f32(tl.load(w.d))
                partial += scale * tl.f32(products)
            tl.store(Y[rows, columns], partial)


@triton.jit(model="weft", alias_groups={"Y": "output"}, strided=("Y",))
def q4_0_prefill_tile(W: tl.View[AMXQ4_0Rows[16], (N, K)], Xq: tl.View[ggml.Q8_0, (M, K)],
                      Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 32,
                      M: tl.constexpr = 32, N: tl.constexpr = 32, BN: tl.constexpr = 32):
    """Specialize the framework's complete tile; K and output strides stay dynamic."""
    tl.static_assert(M == 32 and N == 32)
    compute_rows(W, Xq, Y, M, N, K, BM, BN)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_0_prefill_projection(W: tl.View[AMXQ4_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                            Xq: tl.View[ggml.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)],
                            BM: tl.constexpr = 4, BN: tl.constexpr = 16):
    for row in range(M):
        quantize(X[row], Xq[row], K, False, 4)
    compute_rows(W, Xq, Y, M, N, K, BM, BN)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_0_tile_projection(W: tl.View[AMXQ4_0Rows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                         Xq: tl.View[ggml.Q8_0, (M, K)], Y: tl.View[tl.f32, (M, N)],
                         BM: tl.constexpr = 32, M: tl.constexpr = 32, N: tl.constexpr = 32):
    tl.static_assert(M == 32 and N == 32)
    for row in range(M):
        quantize(X[row], Xq[row], K, False, 4)
    compute_rows(W, Xq, Y, M, N, K, BM, 32)
