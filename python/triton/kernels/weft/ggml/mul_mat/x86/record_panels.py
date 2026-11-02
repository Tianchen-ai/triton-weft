from __future__ import annotations

from triton.integrations import ggml_formats
from ...blocks.quantize import q8_k as quant_q8_k
import triton
import triton.language as tl
from .scaled_records import (
    decode_q2, minimum_q2, decode_q3, decode_q6, decode_iq1_s, decode_iq1_m,
    decode_iq2_xs, decode_iq2_xxs, decode_iq4_xs, decode_iq1,
    iq2_components,
)


@triton.jit
def project(W, X, Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y, Panel, Activation, D, M, N, K,
            BM: tl.constexpr, BN: tl.constexpr, RK: tl.constexpr,
            DECODE: tl.constexpr, STORAGE: tl.constexpr, grid=None, auxiliary=None,
            MINIMUM: tl.constexpr = None, Minima=None, Dmin=None, BIAS: tl.constexpr = 0,
            GROUPS: tl.constexpr = 16, COMPONENTS: tl.constexpr = False,
            INTEGER_CORRECTION: tl.constexpr = False, FACTOR: tl.constexpr = 1, Sums=None,
            CORRECTION_ENGINE: tl.constexpr = "auto"):
    """Reuse scaled records across rows; keep all preparation in the call."""
    tl.static_assert(RK == 256 and RK % GROUPS == 0)
    CORRECTION: tl.constexpr = COMPONENTS or MINIMUM is not None
    quant_q8_k.quantize_matrix(X, Xq)
    first = tl.i1(True)
    with tl.level.blocks(K, extent=RK) as block:
        for row in range(M):
            x = tl.load(Xq[row, block].q)
            tl.store(Activation[row], tl.reshape(x, (RK,), axes=("RK",), order=("K",)))
            if INTEGER_CORRECTION:
                if GROUPS == 8:
                    sums = tl.i16(tl.sum_pairs(tl.load(Xq[row, block].bsum)))
                    sums = tl.reshape(sums, (GROUPS,), axes=("GROUPS",), order=("K",))
                else:
                    grouped = tl.reshape(x, (GROUPS, RK // GROUPS), axes=("GROUPS", "term"), order=("K",))
                    sums = tl.sum(grouped, 1, dtype=tl.i16)
                tl.store(Sums[row], sums)
        with tl.level.cols(N, group=BN) as columns:
            tile = W[columns, block]
            for column in range(BN):
                if column < tl.extent(tile, axis=0):
                    w = tl.load(tile[column])
                    if COMPONENTS:
                        weight, d, minimum = DECODE(w, grid, auxiliary)
                        dmin = d
                    else:
                        weight, d = DECODE(w, grid, auxiliary)
                    tl.store(Panel[column], tl.reshape(STORAGE(weight + tl.i16(BIAS)), (RK,), axes=("RK",), order=("k",)))
                    tl.store(D[column], d)
                    if MINIMUM is not None:
                        minimum, dmin = MINIMUM(w)
                    if CORRECTION:
                        if INTEGER_CORRECTION:
                            minimum = tl.i8(minimum)
                        tl.store(Minima[column], tl.reshape(minimum, (GROUPS,), axes=("GROUPS",), order=("group",)))
                        if not INTEGER_CORRECTION:
                            tl.store(Dmin[column], dmin)
                else:
                    zero = STORAGE(tl.arange(0, RK, axis="RK")) * STORAGE(0)
                    tl.store(Panel[column], zero)
                    tl.store(D[column], tl.f32(0))
                    if CORRECTION:
                        if INTEGER_CORRECTION:
                            tl.store(Minima[column], tl.i8(tl.arange(0, GROUPS, axis="GROUPS")) * tl.i8(0))
                        else:
                            tl.store(Minima[column], tl.i16(tl.arange(0, GROUPS, axis="GROUPS")) * tl.i16(0))
                        if not INTEGER_CORRECTION:
                            tl.store(Dmin[column], tl.f32(0))
            weights = tl.reshape(tl.load(Panel), (BN, RK), axes=("N", "RK"), order=("BN", "RK"))
            d = tl.reshape(tl.load(D), (BN,), axes=("N",), order=("BN",))
            if CORRECTION:
                if not INTEGER_CORRECTION:
                    dmin = tl.reshape(tl.load(Dmin), (BN,), axes=("N",), order=("BN",))
                minimum = tl.reshape(tl.load(Minima), (BN, GROUPS), axes=("N", "group"), order=("BN", "GROUPS"))
            with tl.level.rows(M, group=BM) as rows:
                activation = tl.load(Activation[rows])
                products = tl.reduce_dot(activation, weights, over="RK", acc_dtype=tl.i32)
                x = Xq[rows, block]
                if BIAS:
                    products -= tl.i32(BIAS) * tl.sum(tl.i32(tl.load(x.bsum)), 1)
                xd = tl.load(x.ds)
                if CORRECTION:
                    if INTEGER_CORRECTION:
                        sums = tl.reshape(tl.load(Sums[rows]), (BM, GROUPS), axes=("M", "group"), order=("M", "GROUPS"))
                    elif GROUPS == 16:
                        sums = tl.reshape(tl.load(x.bsum), (BM, GROUPS), axes=("M", "group"), order=("M", "K"))
                    elif GROUPS == 8:
                        sums = tl.reshape(tl.i16(tl.sum_pairs(tl.load(x.bsum))), (BM, GROUPS), axes=("M", "group"), order=("M", "K"))
                    else:
                        grouped = tl.reshape(activation, (BM, GROUPS, RK // GROUPS), axes=("M", "group", "term"), order=("M", "RK"))
                        sums = tl.sum(grouped, 2, dtype=tl.i16)
                    correction = tl.reduce_dot(sums, minimum, over="group", acc_dtype=tl.i32,
                                               engine=CORRECTION_ENGINE)
                    if INTEGER_CORRECTION:
                        products = products * tl.i32(FACTOR) - correction
                result = tl.f32(products) * (xd * d)
                if CORRECTION and not INTEGER_CORRECTION:
                    result -= tl.f32(correction) * (xd * dmin)
                if not first:
                    result += tl.load(Y[rows, columns])
                tl.store(Y[rows, columns], result)
        first = tl.i1(False)
    if first:
        with tl.level.rows(M, group=BM) as rows:
            with tl.level.cols(N, group=BN) as columns:
                zeros = tl.state(tl.f32, [BM, BN], init=0)
                tl.store(Y[rows, columns], zeros)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output", "Minima": "minima", "Dmin": "minimum_scale"})
def q2_k(W: tl.View[ggml_formats.Q2_K, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
         Panel: tl.View[tl.u8, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)],
         D: tl.View[tl.f32, (BN,)], Minima: tl.View[tl.i16, (BN, GROUPS)], Dmin: tl.View[tl.f32, (BN,)],
         BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256, GROUPS: tl.constexpr = 16):
    tl.static_assert(GROUPS == 16)
    # Q2 payload [0,3] times its four-bit scale is in [0,45].
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK,
            decode_q2, tl.u8, MINIMUM=minimum_q2, Minima=Minima, Dmin=Dmin)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def q3_k(W: tl.View[ggml_formats.Q3_K, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
         Panel: tl.View[tl.u8, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
         BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    # Signed 3-bit values times signed 6-bit scales lie in [-124,128].
    # Biasing by124 retains the complete domain in one unsigned byte.
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_q3, tl.u8, BIAS=124)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def q6_k(W: tl.View[ggml_formats.Q6_K, (N, K)], X: tl.View[tl.f32, (M, K)],
         Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
         Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
         BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_q6, tl.i16)


@triton.jit
def decode_iq1_s_byte(w, grid, auxiliary):
    value, scale, delta, d = decode_iq1(w, grid, auxiliary, False, POSITIVE=True)
    weight = (tl.i16(value) - tl.i16(1)) * scale
    minimum = tl.reshape(-delta * scale, (8, 4), axes=("group", "repeat"), order=("entry",))[:, 0]
    return tl.reshape(weight, (256,), axes=("k",), order=("entry", "payload")), d * tl.f32(0.125), minimum


@triton.jit
def decode_iq1_m_byte(w, grid, auxiliary):
    value, scale, delta, d = decode_iq1(w, grid, auxiliary, True, POSITIVE=True)
    weight = (tl.i16(value) - tl.i16(1)) * scale
    minimum = tl.reshape(-delta * scale, (32,), axes=("group",), order=("entry",))
    return tl.reshape(weight, (256,), axes=("k",), order=("entry", "payload")), d * tl.f32(0.125), minimum


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output", "Minima": "correction", "Sums": "group_sums"})
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
          grid: tl.View[ggml_formats.I8X8, (2048, 8)], Y: tl.View[tl.f32, (M, N)],
          Panel: tl.View[tl.i8, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
          Minima: tl.View[tl.i8, (BN, GROUPS)], Sums: tl.View[tl.i16, (M, GROUPS)], BM: tl.constexpr = 16, BN: tl.constexpr = 16,
          RK: tl.constexpr = 256, GROUPS: tl.constexpr = 8):
    tl.static_assert(GROUPS == 8)
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq1_s_byte, tl.i8, grid,
            Minima=Minima, GROUPS=GROUPS, COMPONENTS=True, INTEGER_CORRECTION=True, FACTOR=8, Sums=Sums)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output", "Minima": "correction", "Sums": "group_sums"})
def iq1_m(W: tl.View[ggml_formats.IQ1_M, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
          grid: tl.View[ggml_formats.I8X8, (2048, 8)], f16_bits: tl.View[tl.f32, (65536,)], Y: tl.View[tl.f32, (M, N)],
          Panel: tl.View[tl.i8, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
          Minima: tl.View[tl.i8, (BN, GROUPS)], Sums: tl.View[tl.i16, (M, GROUPS)],
          BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256, GROUPS: tl.constexpr = 32):
    tl.static_assert(GROUPS == 32)
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq1_m_byte, tl.i8, grid, f16_bits,
            Minima=Minima, GROUPS=GROUPS, COMPONENTS=True, INTEGER_CORRECTION=True, FACTOR=8, Sums=Sums,
            CORRECTION_ENGINE="vector")


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq2_xs(W: tl.View[ggml_formats.IQ2_XS, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
           grid: tl.View[ggml_formats.I8X8, (512, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)],
           Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
           BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq2_xs, tl.i16, grid, signs)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq2_xxs(W: tl.View[ggml_formats.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
            grid: tl.View[ggml_formats.I8X8, (256, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)],
            Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
            BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq2_xxs, tl.i16, grid, signs)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq4_xs(W: tl.View[ggml_formats.IQ4_XS, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
           codebook: tl.View[tl.i8, (16,)], Y: tl.View[tl.f32, (M, N)],
           Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
           BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq4_xs, tl.i16, codebook)


@triton.jit
def decode_q4(w, unused, auxiliary):
    lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
    return tl.i16(w.q[lane]) * tl.i16(w.sc[lane // tl.u32(32)]), tl.f32(w.d)


@triton.jit
def decode_q5(w, unused, auxiliary):
    lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
    q = tl.i16(w.q[lane]) | tl.i16(w.qh[lane]) << tl.i16(4)
    return q * tl.i16(w.sc[lane // tl.u32(32)]), tl.f32(w.d)


@triton.jit
def minimum_q45(w):
    group = tl.arange(0, 16, dtype=tl.u32, axis="group")
    return tl.i16(w.m[group // tl.u32(2)]), tl.f32(w.dmin)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output", "Minima": "minima", "Dmin": "minimum_scale"})
def q4_k(W: tl.View[ggml_formats.Q4_K, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
         Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)],
         D: tl.View[tl.f32, (BN,)], Minima: tl.View[tl.i16, (BN, GROUPS)], Dmin: tl.View[tl.f32, (BN,)],
         BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256, GROUPS: tl.constexpr = 16):
    tl.static_assert(GROUPS == 16)
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK,
            decode_q4, tl.i16, MINIMUM=minimum_q45, Minima=Minima, Dmin=Dmin)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output", "Minima": "minima", "Dmin": "minimum_scale"})
def q5_k(W: tl.View[ggml_formats.Q5_K, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
         Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)],
         D: tl.View[tl.f32, (BN,)], Minima: tl.View[tl.i16, (BN, GROUPS)], Dmin: tl.View[tl.f32, (BN,)],
         BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256, GROUPS: tl.constexpr = 16):
    tl.static_assert(GROUPS == 16)
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK,
            decode_q5, tl.i16, MINIMUM=minimum_q45, Minima=Minima, Dmin=Dmin)


@triton.jit
def decode_iq2_s_record(w, grid, auxiliary):
    entry = tl.arange(0, 32, dtype=tl.u16, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
    high = (tl.u16(w.qh[entry // 4]) >> ((entry % 4) * 2)) & tl.u16(3)
    index = tl.u16(w.q[entry]) | (high << 8)
    value = tl.lookup(grid.values, index * 8 + payload, bounds="in_bounds")
    sign = tl.i16(1) - tl.i16(w.signs[entry * 8 + payload]) * tl.i16(2)
    scale = (tl.u16(w.scales[entry // 4]) >> ((entry % 4 // 2) * 4)) & tl.u16(15)
    weight = tl.i16(value) * sign * (tl.i16(scale) * 2 + 1)
    return tl.reshape(weight, (256,), axes=("k",), order=("entry", "payload")), tl.f32(w.d) * 0.125


@triton.jit
def decode_iq3_s_record(w, grid, auxiliary):
    entry = tl.arange(0, 64, dtype=tl.u16, axis="entry")
    payload = tl.arange(0, 4, dtype=tl.u16, axis="payload")
    index = tl.u16(w.q[entry]) | (tl.u16(w.qh[entry]) << 8)
    value = tl.lookup(grid.values, index * 4 + payload, bounds="in_bounds")
    sign = tl.i16(1) - tl.i16(w.signs[entry * 4 + payload]) * tl.i16(2)
    weight = tl.i16(value) * sign * (tl.i16(w.scales[entry // 8]) * 2 + 1)
    return tl.reshape(weight, (256,), axes=("k",), order=("entry", "payload")), tl.f32(w.d)


@triton.jit
def decode_iq2_xs_parity(w, grid, auxiliary):
    value, sign, scale, d = iq2_components(w, grid, None, True, PARITY=True)
    weight = tl.i16(value) * tl.i16(sign) * scale
    return tl.reshape(weight, (256,), axes=("k",), order=("entry", "payload")), d


@triton.jit
def decode_iq2_xxs_parity(w, grid, auxiliary):
    value, sign, scale, d = iq2_components(w, grid, None, False, PARITY=True)
    weight = tl.i16(value) * tl.i16(sign) * scale
    return tl.reshape(weight, (256,), axes=("k",), order=("entry", "payload")), d


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq2_s(W: tl.View[ggml_formats.IQ2_S, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
          grid: tl.View[ggml_formats.I8X8, (1024, 8)], Y: tl.View[tl.f32, (M, N)],
          Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
          BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq2_s_record, tl.i16, grid)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq3_s(W: tl.View[ggml_formats.IQ3_S, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
          grid: tl.View[ggml_formats.I8X4, (512, 4)], Y: tl.View[tl.f32, (M, N)],
          Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
          BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq3_s_record, tl.i16, grid)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq2_xs_parity(W: tl.View[ggml_formats.IQ2_XS, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
                  grid: tl.View[ggml_formats.I8X8, (512, 8)], Y: tl.View[tl.f32, (M, N)],
                  Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
                  BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq2_xs_parity, tl.i16, grid)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq2_xxs_parity(W: tl.View[ggml_formats.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
                   grid: tl.View[ggml_formats.I8X8, (256, 8)], Y: tl.View[tl.f32, (M, N)],
                   Panel: tl.View[tl.i16, (BN, RK)], Activation: tl.View[tl.i8, (M, RK)], D: tl.View[tl.f32, (BN,)],
                   BM: tl.constexpr = 16, BN: tl.constexpr = 16, RK: tl.constexpr = 256):
    project(W, X, Xq, Y, Panel, Activation, D, M, N, K, BM, BN, RK, decode_iq2_xxs_parity, tl.i16, grid)
