from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.derive
def Q4KRows(W: tl.View[ggml.Q4_K, (N, K)], *, rows: tl.static[int]) -> tl.View[Q4KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "sc": 1, "m": 1, "q": 4})


@triton.derive
def Q5KRows(W: tl.View[ggml.Q5_K, (N, K)], *, rows: tl.static[int]) -> tl.View[Q5KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "sc": 6, "m": 6, "qh": 4, "q": 4})


@triton.jit
def affine_metadata(w):
    return tl.load(w.sc), tl.load(w.m), tl.f32(tl.load(w.d)), tl.f32(tl.load(w.dmin))


@triton.jit
def project(W, X, Xq, Y, M, N, K, ROWS: tl.constexpr, HIGH: tl.constexpr,
            META: tl.constexpr, DECODE: tl.constexpr = None, table=None,
            CENTERED: tl.constexpr = False, TERMS: tl.constexpr = 32):
    tl.static_assert(TERMS == 16 or TERMS == 32)
    quant_q8_k.quantize_matrix(X, Xq)
    lane = tl.arange(0, TERMS, dtype=tl.u32, axis="k")
    for row in range(M):
        with tl.level.cols(N, group=ROWS) as columns:
            if tl.extent(W[columns], axis=0) == ROWS:
                result = tl.state(tl.f32, [ROWS], init=0)
                minimum_result = tl.state(tl.f32, [ROWS], init=0)
                with tl.level.blocks(K, extent=256) as block:
                    w, x = W[columns, block], tl.load(Xq[row, block])
                    scales, minima, d, dmin = META(w)
                    integer = tl.state(tl.i32, [ROWS], init=0)
                    minimum = tl.state(tl.i32, [ROWS], init=0)
                    if TERMS == 32:
                        sums = tl.sum_pairs(x.bsum)
                    else:
                        sums = x.bsum
                    for group in tl.static_range(256 // TERMS):
                        q = tl.load(tl.subview(w.q, offsets=(0, group * TERMS), extents=(ROWS, TERMS)))
                        if HIGH:
                            high = tl.load(tl.subview(w.qh, offsets=(0, group * TERMS), extents=(ROWS, TERMS)))
                            q = tl.u6(tl.u8(q) | (tl.u8(high) << 4))
                        if DECODE is not None:
                            q = DECODE(q, table)
                        products = tl.reduce_dot(q, x.q[lane + tl.u32(group * TERMS)], over="k", acc_dtype=tl.i32)
                        integer += products * tl.i32(scales[:, group])
                        minimum += tl.i32(minima[:, group]) * tl.i32(sums[group])
                    if CENTERED:
                        result += (d * tl.f32(x.ds)) * tl.f32(integer - minimum)
                    else:
                        result += (d * tl.f32(x.ds)) * tl.f32(integer)
                        minimum_result += (dmin * tl.f32(x.ds)) * tl.f32(minimum)
                tl.store(Y[row, columns], result - minimum_result)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q4_k(W: tl.View[Q4KRows[ROWS], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
         ROWS: tl.constexpr = 8):
    project(W, X, Xq, Y, M, N, K, ROWS, False, affine_metadata)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q5_k(W: tl.View[Q5KRows[ROWS], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
         ROWS: tl.constexpr = 8):
    project(W, X, Xq, Y, M, N, K, ROWS, True, affine_metadata)


@triton.derive
def IQ4XSRows(W: tl.View[ggml.IQ4_XS, (N, K)], *, rows: tl.static[int]) -> tl.View[IQ4XSRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "scales_h": 2, "scales_l": 4, "q": 4})


@triton.jit
def iq4_metadata(w):
    group = tl.arange(0, 8, dtype=tl.u16, axis="k")
    low = (tl.u16(tl.load(w.scales_l)[:, group // 2]) >> ((group % 2) * 4)) & 15
    high = (tl.u16(tl.load(w.scales_h)) >> (group * 2)) & 3
    scale = tl.i8(low | (high << 4)) - tl.i8(32)
    d = tl.f32(tl.load(w.d))
    return scale, tl.i16(scale) * tl.i16(128), d, d


@triton.jit
def lookup(q, table):
    return tl.lookup(table, tl.u8(q), bounds="in_bounds")


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq4_xs(W: tl.View[IQ4XSRows[ROWS], (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml.Q8_K, (M, K)], codebook: tl.View[tl.i8, (16,)],
            Y: tl.View[tl.f32, (M, N)], ROWS: tl.constexpr = 16):
    table = tl.u8(tl.load(codebook)) + tl.u8(128)
    project(W, X, Xq, Y, M, N, K, ROWS, False, iq4_metadata, lookup, table, True)


@triton.derive
def Q2KRows(W: tl.View[ggml.Q2_K, (N, K)], *, rows: tl.static[int]) -> tl.View[Q2KRows[rows], (N, K)]:
    return tl.interleave(W, rows=rows, field_chunks={"d": 2, "dmin": 2, "scales": 4, "q": 4})


@triton.jit
def q2_metadata(w):
    metadata = tl.load(w.scales)
    return tl.u4(metadata), tl.u4(metadata >> 4), tl.f32(tl.load(w.d)), tl.f32(tl.load(w.dmin))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def q2_k(W: tl.View[Q2KRows[ROWS], (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)], ROWS: tl.constexpr = 16):
    project(W, X, Xq, Y, M, N, K, ROWS, False, q2_metadata, TERMS=16)
