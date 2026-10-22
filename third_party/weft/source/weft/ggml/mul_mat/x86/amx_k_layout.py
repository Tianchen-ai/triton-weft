from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as tl
from weft.ggml.quantize.q8_k_first import q8_k_quantize_first_absolute


@triton.encoding
class AMXQ4_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    q: tl.u4[256] @ tl.grouped(elements=8) @ tl.bit_layers(elements=4, order=tl.lo_first)
    scales: tl.u8[8]
    mins: tl.u8[8]
    d: tl.f16
    dmin: tl.f16


@triton.derive
def AMXQ4_KRows(W: tl.View[AMXQ4_K, (N, K)], *, rows: tl.static[int]) -> tl.View[AMXQ4_KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"q": 4, "scales": 1, "mins": 2, "d": 2, "dmin": 2})


@triton.encoding
class AMXQ6_K:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 256
    low: tl.u4[256] @ tl.grouped(elements=8) @ tl.bit_layers(elements=4, order=tl.lo_first)
    high: tl.u2[256] @ tl.grouped(elements=16) @ tl.bit_layers(elements=4, order=tl.lo_first)
    scales: tl.i8[16]
    d: tl.f16


@triton.derive
def AMXQ6_KRows(W: tl.View[AMXQ6_K, (N, K)], *, rows: tl.static[int]) -> tl.View[AMXQ6_KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"low": 4, "high": 4, "scales": 1, "d": 2})


@triton.jit
def compute_q4_k(W, Xq, Y, N, K, BN: tl.constexpr, COHORT: tl.constexpr = 32):
    tl.static_assert(COHORT % 16 == 0 and BN % COHORT == 0)
    with tl.level.cols(N, group=BN) as columns:
        result = tl.state(tl.f32, [BN], init=0)
        with tl.level.blocks(K, extent=256) as block:
            x = tl.load(Xq[block])
            sums = tl.sum_pairs(x.bsum)
            lane = tl.arange(0, 32, dtype=tl.u32, axis="K")
            for cohort in tl.static_range(0, BN, COHORT):
                w = tl.subview(W[columns, block], offsets=(cohort, 0), extents=(COHORT, 256))
                scales, mins = tl.load(w.scales), tl.load(w.mins)
                for group in tl.static_range(8):
                    weight = tl.load(tl.subview(w.q, offsets=(0, group * 32), extents=(COHORT, 32)))
                    activation = x.q[lane + tl.u32(group * 32)]
                    products = tl.reduce_dot(weight, activation, over="K", acc_dtype=tl.i32)
                    contribution = products * tl.i32(scales[:, group])
                    if group == 0:
                        scaled = contribution
                    else:
                        scaled += contribution
                minimum = tl.reduce_dot(tl.i16(mins), tl.i16(sums), over="K", acc_dtype=tl.i32)
                index = tl.arange(0, COHORT, dtype=tl.u32, axis="N") + tl.u32(cohort)
                partial = result[index]
                partial += (tl.f32(tl.load(w.d)) * tl.f32(x.ds)) * tl.f32(scaled)
                partial += (tl.f32(tl.load(w.dmin)) * tl.f32(x.ds)) * -tl.f32(minimum)
                result[index] = partial
        tl.store(Y[columns], result)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_k_decode(W: tl.View[AMXQ4_KRows[16], (N, K)], Xq: tl.View[ggml.Q8_K, (K,)],
                Y: tl.View[tl.f32, (N,)], BN: tl.constexpr = 128, COHORT: tl.constexpr = 32):
    compute_q4_k(W, Xq, Y, N, K, BN, COHORT)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_k_decode_tile(W: tl.View[AMXQ4_KRows[16], (N, K)], Xq: tl.View[ggml.Q8_K, (M, K)],
                     Y: tl.View[tl.f32, (M, N)], BN: tl.constexpr = 128):
    for row in range(M):
        compute_q4_k(W, Xq[row], Y[row], N, K, BN)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_k_projection(W: tl.View[AMXQ4_KRows[16], (N, K)], X: tl.View[tl.f32, (M, K)],
                     Xq: tl.View[ggml.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)], BN: tl.constexpr = 128):
    for row in range(M):
        q8_k_quantize_first_absolute(X[row], Xq[row])
        compute_q4_k(W, Xq[row], Y[row], N, K, BN)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q6_k_decode(W: tl.View[AMXQ6_KRows[16], (N, K)], Xq: tl.View[ggml.Q8_K, (K,)],
                Y: tl.View[tl.f32, (N,)], BN: tl.constexpr = 64):
    with tl.level.cols(N, group=BN) as columns:
        result = tl.state(tl.f32, [BN], init=0)
        with tl.level.blocks(K, extent=256) as block:
            w, x = W[columns, block], tl.load(Xq[block])
            scales = tl.load(w.scales)
            lane = tl.arange(0, 16, dtype=tl.u32, axis="K")
            for group in tl.static_range(16):
                low = tl.load(tl.subview(w.low, offsets=(0, group * 16), extents=(BN, 16)))
                high = tl.load(tl.subview(w.high, offsets=(0, group * 16), extents=(BN, 16)))
                weight = tl.u8(low) | (tl.u8(high) << tl.u8(4))
                activation = x.q[lane + tl.u32(group * 16)]
                products = tl.reduce_dot(weight, activation, over="K", acc_dtype=tl.i32)
                centered = products - tl.i32(x.bsum[group]) * tl.i32(32)
                contribution = centered * tl.i32(scales[:, group])
                if group == 0:
                    scaled = contribution
                else:
                    scaled += contribution
            result += (tl.f32(tl.load(w.d)) * tl.f32(x.ds)) * tl.f32(scaled)
        tl.store(Y[columns], result)


@triton.jit(model="weft", alias_groups={"Y": "output"}, strided=("Y",))
def q4_k_prefill_tile(W: tl.View[AMXQ4_KRows[16], (N, K)], Xq: tl.View[ggml.Q8_K, (M, K)],
                      Y: tl.View[tl.f32, (M, N)], M: tl.constexpr = 32, N: tl.constexpr = 32,
                      BM: tl.constexpr = 16, BN: tl.constexpr = 32):
    tl.static_assert(M == 32 and N == 32)
    tl.static_assert((BM == 16 or BM == 32) and (BN == 16 or BN == 32))
    with tl.level.cols(N, group=BN) as columns:
        with tl.level.rows(M, group=BM) as rows:
            result = tl.state(tl.f32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=256) as block:
                w, x = W[columns, block], Xq[rows, block]
                scales = tl.load(w.scales)
                for group in tl.static_range(8):
                    weight = tl.i8(tl.load(tl.subview(w.q, offsets=(0, group * 32), extents=(BN, 32))))
                    activation = tl.load(tl.subview(x.q, offsets=(0, group * 32), extents=(BM, 32)))
                    products = tl.reduce_dot(activation, weight, over="K", acc_dtype=tl.i32)
                    contribution = products * tl.i32(scales[:, group])
                    if group == 0:
                        scaled = contribution
                    else:
                        scaled += contribution
                sums = tl.i16(tl.sum_pairs(tl.load(x.bsum)))
                minimum = tl.reduce_dot(sums, tl.i16(tl.load(w.mins)), over="K", acc_dtype=tl.i32)
                activation_scale = tl.f32(tl.load(x.ds))
                result += tl.f32(scaled) * (activation_scale * tl.f32(tl.load(w.d)))
                result += tl.f32(minimum) * (-activation_scale * tl.f32(tl.load(w.dmin)))
            tl.store(Y[rows, columns], result)


@triton.jit(model="weft", alias_groups={"Y": "output"}, strided=("Y",))
def q6_k_prefill_tile(W: tl.View[AMXQ6_KRows[16], (N, K)], Xq: tl.View[ggml.Q8_K, (M, K)],
                      Y: tl.View[tl.f32, (M, N)], M: tl.constexpr = 32, N: tl.constexpr = 32,
                      BM: tl.constexpr = 16, BN: tl.constexpr = 32):
    tl.static_assert(M == 32 and N == 32)
    tl.static_assert((BM == 16 or BM == 32) and (BN == 16 or BN == 32))
    with tl.level.cols(N, group=BN) as columns:
        with tl.level.rows(M, group=BM) as rows:
            result = tl.state(tl.f32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=256) as block:
                w, x = W[columns, block], Xq[rows, block]
                scales = tl.load(w.scales)
                for group in tl.static_range(16):
                    low = tl.load(tl.subview(w.low, offsets=(0, group * 16), extents=(BN, 16)))
                    high = tl.load(tl.subview(w.high, offsets=(0, group * 16), extents=(BN, 16)))
                    weight = tl.i8(tl.u8(low) | (tl.u8(high) << tl.u8(4))) - tl.i8(32)
                    activation = tl.load(tl.subview(x.q, offsets=(0, group * 16), extents=(BM, 16)))
                    products = tl.reduce_dot(activation, weight, over="K", acc_dtype=tl.i32)
                    contribution = products * tl.i32(scales[:, group])
                    if group == 0:
                        scaled = contribution
                    else:
                        scaled += contribution
                result += tl.f32(scaled) * (tl.f32(tl.load(x.ds)) * tl.f32(tl.load(w.d)))
            tl.store(Y[rows, columns], result)


@triton.jit(model="weft", alias_groups={"Y": "output"}, strided=("Y",))
def q6_k_prefill_scaled(W: tl.View[AMXQ6_KRows[16], (N, K)], Xq: tl.View[ggml.Q8_K, (M, K)],
                        Y: tl.View[tl.f32, (M, N)], M: tl.constexpr = 32, N: tl.constexpr = 32,
                        BM: tl.constexpr = 16, BN: tl.constexpr = 16):
    tl.static_assert(M == 32 and N == 32)
    with tl.level.cols(N, group=BN) as columns:
        with tl.level.rows(M, group=BM) as rows:
            result = tl.state(tl.f32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=256) as block:
                w, x = W[columns, block], Xq[rows, block]
                lane = tl.arange(0, 256, dtype=tl.u32, axis="K")
                bits = tl.u8(tl.load(w.low)) | (tl.u8(tl.load(w.high)) << tl.u8(4))
                weight = tl.i16(bits) - tl.i16(32)
                scales = tl.i16(tl.load(w.scales)[:, lane // tl.u32(16)])
                # The full I6 x I8 scale product fits I16. The author chooses
                # this exact integer realization before the closed contraction.
                products = tl.reduce_dot(tl.load(x.q), weight * scales, over="K", acc_dtype=tl.i32)
                result += tl.f32(products) * (tl.f32(tl.load(x.ds)) * tl.f32(tl.load(w.d)))
            tl.store(Y[rows, columns], result)
