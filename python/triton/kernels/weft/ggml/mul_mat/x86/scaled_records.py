from __future__ import annotations

from triton.integrations import ggml_formats
from ...blocks.quantize import q8_k as quant_q8_k
import triton
import triton.language as tl
from ...vec_dot.x86.signs import even_parity_signs
from ...dequantize.radix import ternary_record
from .codebook import project_indexed_tiles


@triton.jit
def decode_iq1(w, grid, f16_bits, MEDIUM: tl.constexpr, GROUPED: tl.constexpr = False,
               POSITIVE: tl.constexpr = False):
    entry = tl.arange(0, 32, dtype=tl.u16, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
    if MEDIUM:
        sc0 = tl.u16(w.scales[0]) | tl.u16(w.scales[1]) << tl.u16(8)
        sc1 = tl.u16(w.scales[2]) | tl.u16(w.scales[3]) << tl.u16(8)
        sc2 = tl.u16(w.scales[4]) | tl.u16(w.scales[5]) << tl.u16(8)
        sc3 = tl.u16(w.scales[6]) | tl.u16(w.scales[7]) << tl.u16(8)
        bits = (sc0 >> tl.u16(12)) | ((sc1 >> tl.u16(8)) & tl.u16(240))
        bits = bits | ((sc2 >> tl.u16(4)) & tl.u16(3840)) | (sc3 & tl.u16(61440))
        d = tl.lookup(f16_bits, bits, bounds="in_bounds")
        metadata = tl.u16(w.qh[entry // tl.u16(2)]) >> (entry % tl.u16(2) * tl.u16(4))
        high = metadata & tl.u16(7)
        delta = tl.i16(1) - tl.i16((metadata >> tl.u16(3)) & tl.u16(1)) * tl.i16(2)
        group = entry // tl.u16(4)
        sc = tl.u16(w.scales[group // tl.u16(2) * tl.u16(2)])
        sc = sc | tl.u16(w.scales[group // tl.u16(2) * tl.u16(2) + tl.u16(1)]) << tl.u16(8)
        shift = group % tl.u16(2) * tl.u16(6) + entry % tl.u16(4) // tl.u16(2) * tl.u16(3)
        scale = tl.i16((sc >> shift) & tl.u16(7)) * tl.i16(2) + tl.i16(1)
    else:
        d = tl.f32(w.d)
        metadata = tl.u16(w.qh[entry // tl.u16(4)])
        high = (metadata >> (entry % tl.u16(4) * tl.u16(3))) & tl.u16(7)
        delta = tl.i16(1) - tl.i16(metadata >> tl.u16(15)) * tl.i16(2)
        scale = tl.i16((metadata >> tl.u16(12)) & tl.u16(7)) * tl.i16(2) + tl.i16(1)
    index = tl.u16(w.q[entry]) | high << tl.u16(8)
    value = tl.lookup(grid.values, index * tl.u16(8) + payload, bounds="in_bounds")
    if POSITIVE:
        return tl.u2(value + tl.i8(1)), scale, delta, d
    # The fractional IQ1 correction is an exact eighth. Keeping that factor
    # outside the integer dot gives one signed-word program for both formats.
    if GROUPED:
        return tl.i8(value) * tl.i8(8) + tl.i8(delta), scale, d * tl.f32(0.125)
    weight = (tl.i16(value) * tl.i16(8) + delta) * scale
    return tl.reshape(weight, (256,), axes=("k",), order=("entry", "payload")), d * tl.f32(0.125)


@triton.jit
def iq2_components(w, grid, signs, XS: tl.constexpr, PARITY: tl.constexpr = False):
    entry = tl.arange(0, 32, dtype=tl.u16, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
    if XS:
        word = tl.u16(w.q[entry])
        index = word & tl.u16(511)
        sign_index = word >> tl.u16(9)
        metadata = tl.u16(w.scales[entry // tl.u16(4)])
        scale = (metadata >> (entry % tl.u16(4) // tl.u16(2) * tl.u16(4))) & tl.u16(15)
    else:
        group = entry // tl.u16(4)
        metadata = tl.u32(w.q[group * tl.u16(4) + tl.u16(2)]) | tl.u32(w.q[group * tl.u16(4) + tl.u16(3)]) << tl.u32(16)
        word = tl.u16(w.q[group * tl.u16(4) + entry % tl.u16(4) // tl.u16(2)])
        index = (word >> (entry % tl.u16(2) * tl.u16(8))) & tl.u16(255)
        sign_index = (metadata >> (entry % tl.u16(4) * tl.u16(7))) & tl.u16(127)
        scale = metadata >> tl.u16(28)
    value = tl.lookup(grid.values, index * tl.u16(8) + payload, bounds="in_bounds")
    if PARITY:
        sign = even_parity_signs(sign_index, payload)
    else:
        sign = tl.lookup(signs.values, sign_index * tl.u16(8) + payload, bounds="in_bounds")
    return value, sign, tl.i16(scale) * tl.i16(2) + tl.i16(1), tl.f32(w.d) * tl.f32(0.125)


@triton.jit
def decode_iq2(w, grid, signs, XS: tl.constexpr, GROUPED: tl.constexpr = False):
    value, sign, scale, d = iq2_components(w, grid, signs, XS)
    if GROUPED:
        signed = tl.where(sign < tl.i8(0), -value, value)
        return signed, scale, d
    weight = tl.i16(value) * tl.i16(sign) * scale
    return tl.reshape(weight, (256,), axes=("k",), order=("entry", "payload")), d


@triton.jit
def decode_iq1_s(w, grid, auxiliary):
    return decode_iq1(w, grid, auxiliary, False)


@triton.jit
def decode_iq1_m(w, grid, auxiliary):
    return decode_iq1(w, grid, auxiliary, True)


@triton.jit
def decode_iq2_xs(w, grid, auxiliary):
    return decode_iq2(w, grid, auxiliary, True)


@triton.jit
def decode_iq2_xxs(w, grid, auxiliary):
    return decode_iq2(w, grid, auxiliary, False)


@triton.jit
def grouped_iq1_s(w, grid, auxiliary):
    weight, scale, d = decode_iq1(w, grid, auxiliary, False, True)
    scale = tl.reshape(scale, (16, 2), axes=("group", "pair"), order=("entry",))
    weight = tl.reshape(weight, (16, 2, 8), axes=("group", "entry", "payload"), order=("entry", "payload"))
    return weight, scale[:, 0], d


@triton.jit
def grouped_iq1_m(w, grid, auxiliary):
    weight, scale, d = decode_iq1(w, grid, auxiliary, True, True)
    scale = tl.reshape(scale, (16, 2), axes=("group", "pair"), order=("entry",))
    weight = tl.reshape(weight, (16, 2, 8), axes=("group", "entry", "payload"), order=("entry", "payload"))
    return weight, scale[:, 0], d


@triton.jit
def grouped_iq2_xs(w, grid, auxiliary):
    weight, scale, d = decode_iq2(w, grid, auxiliary, True, True)
    scale = tl.reshape(scale, (16, 2), axes=("group", "pair"), order=("entry",))
    weight = tl.reshape(weight, (16, 2, 8), axes=("group", "entry", "payload"), order=("entry", "payload"))
    return weight, scale[:, 0], d


@triton.jit
def grouped_iq2_xxs(w, grid, auxiliary):
    weight, scale, d = decode_iq2(w, grid, auxiliary, False, True)
    scale = tl.reshape(scale, (16, 2), axes=("group", "pair"), order=("entry",))
    weight = tl.reshape(weight, (16, 2, 8), axes=("group", "entry", "payload"), order=("entry", "payload"))
    return weight, scale[:, 0], d


@triton.jit
def project(W, X, Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid, auxiliary, Y,
            Panel, Activation, D, M, N, BM: tl.constexpr, BN: tl.constexpr,
            PK: tl.constexpr, PAIR: tl.constexpr, DECODE: tl.constexpr,
            AFFINE: tl.constexpr = False, MINIMUM: tl.constexpr = None,
            Minima=None, Dmin=None):
    """Decode each weight record once per invocation and reuse across row tiles.

    The scaled weights fit signed i16 and each complete record dot fits i32.
    Quantization, panel stores and output initialization are invocation work.
    """
    tl.static_assert(PK == 128 and PAIR == 2)
    quant_q8_k.quantize_matrix(X, Xq)
    first = tl.i1(True)
    with tl.level.blocks(K, extent=256) as block:
        for row in range(M):
            x = tl.load(Xq[row, block])
            values = tl.reshape(tl.i16(x.q), (PK, PAIR), axes=("PK", "PAIR"), order=("K",))
            tl.store(Activation[:, row, :], values)
        with tl.level.cols(N, group=BN) as columns:
            tile = W[columns, block]
            for column in range(BN):
                if column < tl.extent(tile, axis=0):
                    w = tl.load(tile[column])
                    values, d = DECODE(w, grid, auxiliary)
                    pairs = tl.reshape(values, (PK, PAIR), axes=("PK", "PAIR"), order=("k",))
                    tl.store(Panel[:, column, :], pairs)
                    tl.store(D[column], d)
                    if AFFINE:
                        minimum, dmin = MINIMUM(w)
                        tl.store(Minima[column], tl.reshape(minimum, (16,), axes=("GROUPS",), order=("group",)))
                        tl.store(Dmin[column], dmin)
                else:
                    zeros = tl.i16(tl.arange(0, PK, axis="PK")) * tl.i16(0) + tl.i16(tl.arange(0, PAIR, axis="PAIR")) * tl.i16(0)
                    tl.store(Panel[:, column, :], zeros)
                    tl.store(D[column], tl.f32(0))
                    if AFFINE:
                        tl.store(Minima[column], tl.i16(tl.arange(0, 16, axis="GROUPS")) * tl.i16(0))
                        tl.store(Dmin[column], tl.f32(0))
            d = tl.reshape(tl.load(D), (BN,), axes=("N",), order=("BN",))
            with tl.level.rows(M, group=BM) as rows:
                acc = tl.state(tl.i32, [BM, BN], init=0)
                for pair in range(PK):
                    weights = tl.reshape(tl.load(Panel[pair]), (BN, PAIR), axes=("N", "pair"), order=("BN", "PAIR"))
                    activation = tl.reshape(tl.load(Activation[pair, rows]), (BM, PAIR), axes=("M", "pair"), order=("M", "PAIR"))
                    product = tl.mac_pairs(activation, weights, into=tl.i32)
                    acc += tl.reshape(product, (BM, BN), axes=("M", "N"), order=("M", "N", "pair"))
                xd = tl.load(Xq[rows, block].ds)
                result = tl.f32(acc) * (xd * d)
                if AFFINE:
                    minimum = tl.reshape(tl.load(Minima), (BN, 16), axes=("N", "group"), order=("BN", "GROUPS"))
                    sums = tl.reshape(tl.load(Xq[rows, block].bsum), (BM, 16), axes=("M", "group"), order=("M", "K"))
                    offset = tl.reduce_dot(sums, minimum, over="group", acc_dtype=tl.i32)
                    md = tl.reshape(tl.load(Dmin), (BN,), axes=("N",), order=("BN",))
                    result -= tl.f32(offset) * (xd * md)
                if not first:
                    result += tl.load(Y[rows, columns])
                tl.store(Y[rows, columns], result)
        first = tl.i1(False)
    if first:
        with tl.level.rows(M, group=BM) as rows:
            with tl.level.cols(N, group=BN) as columns:
                zeros = tl.state(tl.f32, [BM, BN], init=0)
                tl.store(Y[rows, columns], zeros)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
          Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.i16, (PK, BN, PAIR)],
          Activation: tl.View[tl.i16, (PK, M, PAIR)], D: tl.View[tl.f32, (BN,)],
          BM: tl.constexpr = 4, BN: tl.constexpr = 32, PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, grid, None, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_iq1_s)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq1_m(W: tl.View[ggml_formats.IQ1_M, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
          f16_bits: tl.View[tl.f32, (65536,)], Y: tl.View[tl.f32, (M, N)],
          Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
          D: tl.View[tl.f32, (BN,)], BM: tl.constexpr = 4, BN: tl.constexpr = 32,
          PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, grid, f16_bits, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_iq1_m)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq2_xs(W: tl.View[ggml_formats.IQ2_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (512, 8)],
           signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)],
           Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
           D: tl.View[tl.f32, (BN,)], BM: tl.constexpr = 4, BN: tl.constexpr = 32,
           PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, grid, signs, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_iq2_xs)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq2_xxs(W: tl.View[ggml_formats.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (256, 8)],
            signs: tl.View[ggml_formats.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)],
            Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
            D: tl.View[tl.f32, (BN,)], BM: tl.constexpr = 4, BN: tl.constexpr = 32,
            PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, grid, signs, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_iq2_xxs)

@triton.jit
def decode_q3(w, unused, auxiliary):
    lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
    group = lane // tl.u32(16)
    scale = (tl.i16(w.scale_low[group]) | tl.i16(w.scale_high[group]) << tl.u16(4)) - tl.i16(32)
    q = tl.i16(w.q[lane]) + tl.i16(w.hmask[lane]) * tl.i16(4) - tl.i16(4)
    return q * scale, tl.f32(w.d)


@triton.jit
def grouped_q3(w, unused, auxiliary):
    group = tl.arange(0, 16, dtype=tl.u32, axis="group")
    entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
    lane = group * tl.u32(16) + entry * tl.u32(8) + payload
    scale = (tl.i32(w.scale_low[group]) | tl.i32(w.scale_high[group]) << tl.u32(4)) - tl.i32(32)
    q = tl.i8(w.q[lane]) + tl.i8(w.hmask[lane]) * tl.i8(4) - tl.i8(4)
    return q, scale, tl.f32(w.d)


@triton.jit
def decode_q6(w, unused, auxiliary):
    lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
    q = (tl.i16(w.ql[lane]) | tl.i16(w.qh[lane]) << tl.u16(4)) - tl.i16(32)
    return q * tl.i16(w.scales[lane // tl.u32(16)]), tl.f32(w.d)


@triton.jit
def decode_q2(w, unused, auxiliary):
    lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
    scale = tl.i16(tl.u8(w.scales[lane // tl.u32(16)]) & tl.u8(15))
    return tl.i16(w.q[lane]) * scale, tl.f32(w.d)


@triton.jit
def minimum_q2(w):
    group = tl.arange(0, 16, dtype=tl.u32, axis="group")
    return tl.i16(tl.u8(w.scales[group]) >> tl.u8(4)), tl.f32(w.dmin)


@triton.jit
def decode_iq4_xs(w, codebook, auxiliary):
    lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
    group = lane // tl.u32(32)
    low = (tl.u32(w.scales_l[group // tl.u32(2)]) >> (group % tl.u32(2) * tl.u32(4))) & tl.u32(15)
    high = (tl.u32(w.scales_h) >> (group * tl.u32(2))) & tl.u32(3)
    scale = tl.i16(low | high << tl.u32(4)) - tl.i16(32)
    value = tl.lookup(codebook, tl.u8(w.q[lane]), bounds="in_bounds")
    return tl.i16(value) * scale, tl.f32(w.d)


@triton.jit
def grouped_iq4_xs(w, codebook, auxiliary):
    group = tl.arange(0, 8, dtype=tl.u32, axis="group")
    entry = tl.arange(0, 8, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 4, dtype=tl.u32, axis="payload")
    lane = group * tl.u32(32) + entry * tl.u32(4) + payload
    low = (tl.u32(w.scales_l[group // tl.u32(2)]) >> (group % tl.u32(2) * tl.u32(4))) & tl.u32(15)
    high = (tl.u32(w.scales_h) >> (group * tl.u32(2))) & tl.u32(3)
    scale = tl.i32(low | high << tl.u32(4)) - tl.i32(32)
    value = tl.lookup(codebook, tl.u8(w.q[lane]), bounds="in_bounds")
    return value, scale, tl.f32(w.d)


@triton.jit
def decode_tq1(w, powers, auxiliary):
    q = tl.i16(ternary_record(w)) - tl.i16(1)
    return q, tl.f32(w.d)


@triton.jit
def decode_tq2(w, unused, auxiliary):
    return tl.reshape(tl.i16(w.q) - tl.i16(1), (256,), axes=("k",), order=("K",)), tl.f32(w.d)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output", "Minima": "minima", "Dmin": "minimum_scale"})
def q2_k(W: tl.View[ggml_formats.Q2_K, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
            Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
            D: tl.View[tl.f32, (BN,)], Minima: tl.View[tl.i16, (BN, GROUPS)], Dmin: tl.View[tl.f32, (BN,)],
            GROUPS: tl.constexpr = 16, BM: tl.constexpr = 4, BN: tl.constexpr = 32,
            PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, None, None, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_q2, True, minimum_q2, Minima, Dmin)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def q3_k(W: tl.View[ggml_formats.Q3_K, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
            Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
            D: tl.View[tl.f32, (BN,)], BM: tl.constexpr = 4, BN: tl.constexpr = 32,
            PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, None, None, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_q3)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def q6_k(W: tl.View[ggml_formats.Q6_K, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
            Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
            D: tl.View[tl.f32, (BN,)], BM: tl.constexpr = 4, BN: tl.constexpr = 32,
            PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, None, None, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_q6)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def iq4_xs(W: tl.View[ggml_formats.IQ4_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml_formats.Q8_K, (M, K)], codebook: tl.View[tl.i8, (16,)], Y: tl.View[tl.f32, (M, N)],
            Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
            D: tl.View[tl.f32, (BN,)], BM: tl.constexpr = 4, BN: tl.constexpr = 32,
            PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, codebook, None, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_iq4_xs)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def tq1_0(W: tl.View[ggml_formats.TQ1_0, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml_formats.Q8_K, (M, K)], powers: tl.View[tl.u32, (5,)], Y: tl.View[tl.f32, (M, N)],
            Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
            D: tl.View[tl.f32, (BN,)], BM: tl.constexpr = 4, BN: tl.constexpr = 32,
            PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, powers, None, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_tq1)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Panel": "weights", "Activation": "activation", "D": "scale", "Y": "output"})
def tq2_0(W: tl.View[ggml_formats.TQ2_0, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
          Panel: tl.View[tl.i16, (PK, BN, PAIR)], Activation: tl.View[tl.i16, (PK, M, PAIR)],
          D: tl.View[tl.f32, (BN,)], BM: tl.constexpr = 4, BN: tl.constexpr = 32,
          PK: tl.constexpr = 128, PAIR: tl.constexpr = 2):
    project(W, X, Xq, None, None, Y, Panel, Activation, D, M, N, BM, BN, PK, PAIR, decode_tq2)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def iq1_s_bytes(W: tl.View[ggml_formats.IQ1_S, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
                  Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                  Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                  Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                  XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                  BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 16,
                  WORDS: tl.constexpr = 4, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 16 and WORDS == 4 and PACK == 4)
    project_indexed_tiles(W, X, Xq, grid, Y, Panel, Scales, D, Activation, Sums, XScale, Output,
                          N, AK, BM, BN, GROUPS, WORDS, PACK, True, signs=None, DECODE=grouped_iq1_s)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def iq1_m_bytes(W: tl.View[ggml_formats.IQ1_M, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (2048, 8)],
                  f16_bits: tl.View[tl.f32, (65536,)],
                  Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                  Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                  Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                  XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                  BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 16,
                  WORDS: tl.constexpr = 4, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 16 and WORDS == 4 and PACK == 4)
    project_indexed_tiles(W, X, Xq, grid, Y, Panel, Scales, D, Activation, Sums, XScale, Output,
                          N, AK, BM, BN, GROUPS, WORDS, PACK, True, signs=f16_bits, DECODE=grouped_iq1_m)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def iq2_xs_bytes(W: tl.View[ggml_formats.IQ2_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (512, 8)],
                  signs: tl.View[ggml_formats.I8X8, (128, 8)],
                  Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                  Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                  Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                  XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                  BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 16,
                  WORDS: tl.constexpr = 4, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 16 and WORDS == 4 and PACK == 4)
    project_indexed_tiles(W, X, Xq, grid, Y, Panel, Scales, D, Activation, Sums, XScale, Output,
                          N, AK, BM, BN, GROUPS, WORDS, PACK, True, signs=signs, DECODE=grouped_iq2_xs)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def iq2_xxs_bytes(W: tl.View[ggml_formats.IQ2_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml_formats.Q8_K, (M, K)], grid: tl.View[ggml_formats.I8X8, (256, 8)],
                  signs: tl.View[ggml_formats.I8X8, (128, 8)],
                  Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                  Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                  Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                  XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                  BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 16,
                  WORDS: tl.constexpr = 4, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 16 and WORDS == 4 and PACK == 4)
    project_indexed_tiles(W, X, Xq, grid, Y, Panel, Scales, D, Activation, Sums, XScale, Output,
                          N, AK, BM, BN, GROUPS, WORDS, PACK, True, signs=signs, DECODE=grouped_iq2_xxs)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def q3_k_bytes(W: tl.View[ggml_formats.Q3_K, (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
                 Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                 Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                 Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                 XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                 BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 16,
                 WORDS: tl.constexpr = 4, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 16 and WORDS == 4 and PACK == 4)
    project_indexed_tiles(W, X, Xq, None, Y, Panel, Scales, D, Activation, Sums, XScale, Output,
                          N, AK, BM, BN, GROUPS, WORDS, PACK, True, DECODE=grouped_q3)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def iq4_xs_bytes(W: tl.View[ggml_formats.IQ4_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
                 Xq: tl.View[ggml_formats.Q8_K, (M, K)], codebook: tl.View[tl.i8, (16,)], Y: tl.View[tl.f32, (M, N)],
                 Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                 Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                 Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                 XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                 BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 8,
                 WORDS: tl.constexpr = 8, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 8 and WORDS == 8 and PACK == 4)
    project_indexed_tiles(W, X, Xq, codebook, Y, Panel, Scales, D, Activation, Sums, XScale, Output,
                          N, AK, BM, BN, GROUPS, WORDS, PACK, False, DECODE=grouped_iq4_xs)
