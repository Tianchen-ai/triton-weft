from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_0 as quant_q8_0
import kernels.quantize.q8_k as quant_q8_k
import triton
import triton.language as tl
from weft.ggml.vec_dot.x86.codebook import compute_iq4_nl, compute_iq4_xs, compute_mxfp4
from weft.ggml.vec_dot.x86.indexed import compute_iq2_s, compute_iq3_s, compute_iq3_xxs, decode_iq3_xxs


@triton.jit
def project(W, X, Xq, codebook, scale, Y, M, N,
            QUANTIZE: tl.constexpr, DOT: tl.constexpr, WITH_SCALE: tl.constexpr):
    """Include activation quantization and all output rows in the native call."""
    QUANTIZE(X, Xq)
    for row in range(M):
        for column in range(N):
            if WITH_SCALE:
                value = DOT(W[column], Xq[row], codebook, scale)
            else:
                value = DOT(W[column], Xq[row], codebook)
            tl.store(Y[row, column], value)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq4_nl(W: tl.View[ggml.IQ4_NL, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
           Y: tl.View[tl.f32, (M, N)]):
    project(W, X, Xq, codebook, None, Y, M, N,
            quant_q8_0.quantize_matrix, compute_iq4_nl, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq4_xs(W: tl.View[ggml.IQ4_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
           Xq: tl.View[ggml.Q8_K, (M, K)], codebook: tl.View[tl.i8, (16,)],
           Y: tl.View[tl.f32, (M, N)]):
    project(W, X, Xq, codebook, None, Y, M, N,
            quant_q8_k.quantize_matrix, compute_iq4_xs, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def mxfp4(W: tl.View[ggml.MXFP4, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
          scale: tl.View[tl.f32, (256,)], Y: tl.View[tl.f32, (M, N)]):
    project(W, X, Xq, codebook, scale, Y, M, N,
            quant_q8_0.quantize_matrix, compute_mxfp4, True)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_s(W: tl.View[ggml.IQ2_S, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X8, (1024, 8)],
          Y: tl.View[tl.f32, (M, N)]):
    project(W, X, Xq, grid, None, Y, M, N,
            quant_q8_k.quantize_matrix, compute_iq2_s, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq3_s(W: tl.View[ggml.IQ3_S, (N, K)], X: tl.View[tl.f32, (M, K)],
          Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X4, (512, 4)],
          Y: tl.View[tl.f32, (M, N)]):
    project(W, X, Xq, grid, None, Y, M, N,
            quant_q8_k.quantize_matrix, compute_iq3_s, False)


@triton.jit
def project_scaled_blocks(W, X, Xq: tl.View[ggml.Q8_K, (M, K)], grid, Y, N,
                          IQ2: tl.constexpr, XXS: tl.constexpr = False, signs=None):
    """Decode a scaled integer weight block once for all activation rows.

    Scale products fit i16 for these codebooks; the 256-element dot fits i32.
    All quantization, decoding and output initialization remain in this call.
    """
    quant_q8_k.quantize_matrix(X, Xq)
    if IQ2:
        group = tl.arange(0, 16, dtype=tl.u32, axis="group")
        entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
        code = group * tl.u32(2) + entry
        position = group * tl.u32(16) + entry * tl.u32(8) + payload
    else:
        group = tl.arange(0, 8, dtype=tl.u32, axis="group")
        entry = tl.arange(0, 8, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 4, dtype=tl.u32, axis="payload")
        code = group * tl.u32(8) + entry
        position = group * tl.u32(32) + entry * tl.u32(4) + payload
    for column in range(N):
        for row in range(M):
            tl.store(Y[row, column], tl.f32(0))
        with tl.level.blocks(K, extent=256) as block:
            w = tl.load(W[column, block])
            if IQ2:
                high = (tl.u32(w.qh[code // tl.u32(4)]) >> (code % tl.u32(4) * tl.u32(2))) & tl.u32(3)
                index = tl.u32(w.q[code]) | (high << tl.u32(8))
                weight = tl.lookup(grid.values, index * tl.u32(8) + payload, bounds="in_bounds")
                metadata = tl.u32(w.scales[group // tl.u32(2)])
                scale = tl.i16((metadata >> (group % tl.u32(2) * tl.u32(4))) & tl.u32(15)) * tl.i16(2) + tl.i16(1)
                d = tl.f32(w.d) * tl.f32(0.125)
            elif XXS:
                weight, scale, position = decode_iq3_xxs(w, grid, signs)
                d = tl.f32(w.d) * tl.f32(0.25)
            else:
                index = tl.u32(w.q[code]) | (tl.u32(w.qh[code]) << tl.u32(8))
                weight = tl.lookup(grid.values, index * tl.u32(4) + payload, bounds="in_bounds")
                scale = tl.i16(w.scales[group]) * tl.i16(2) + tl.i16(1)
                d = tl.f32(w.d)
            if not XXS:
                sign = tl.i8(1) - tl.i8(w.signs[position]) * tl.i8(2)
                weight = weight * sign
            for row in range(M):
                x = tl.load(Xq[row, block])
                products = tl.mac_groups(weight, x.q[position], n=4, into=tl.i32)
                scaled = products * tl.i32(scale)
                total = tl.sum(tl.sum(tl.sum(scaled, 2), 1), 0)
                prior = tl.load(Y[row, column])
                tl.store(Y[row, column], prior + (d * tl.f32(x.ds)) * tl.f32(total))


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq2_s_prefill(W: tl.View[ggml.IQ2_S, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X8, (1024, 8)],
                  Y: tl.View[tl.f32, (M, N)]):
    project_scaled_blocks(W, X, Xq, grid, Y, N, True)


@triton.jit
def project_indexed_tiles(W, X, Xq: tl.View[ggml.Q8_K, (M, K)], grid, Y, Panel, Scales, D,
                          Activation, Sums, XScale, Output, N, AK: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, GROUPS: tl.constexpr, WORDS: tl.constexpr,
                          PACK: tl.constexpr, IQ2: tl.constexpr, XXS: tl.constexpr = False, signs=None,
                          AFFINE: tl.constexpr = False, HIGH: tl.constexpr = False, Minima=None, Dmin=None,
                          DECODE: tl.constexpr = None):
    """Pack each activation slab once, then reuse it across weight tiles."""
    tl.static_assert(AK == GROUPS * WORDS)
    quant_q8_k.quantize_matrix(X, Xq)
    group = tl.arange(0, GROUPS, dtype=tl.u32, axis="group")
    if IQ2:
        entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
        code = group * tl.u32(2) + entry
        position = group * tl.u32(16) + entry * tl.u32(8) + payload
    else:
        entry = tl.arange(0, 8, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 4, dtype=tl.u32, axis="payload")
        code = group * tl.u32(8) + entry
        position = group * tl.u32(32) + entry * tl.u32(4) + payload
    with tl.level.cols(N, group=BN) as columns:
        with tl.level.rows(M, group=BM) as rows:
            empty = tl.state(tl.f32, [BN, BM], init=0)
            tl.store(Output[columns, rows], empty)
    with tl.level.blocks(K, extent=256) as block:
        with tl.level.rows(M, group=BM) as rows:
            x = Xq[rows, block]
            activation = tl.reshape(tl.load(x.q), (BM, AK, PACK), axes=("M", "AK", "PACK"), order=("M", "K"))
            activation = tl.reshape(activation, (AK, BM, PACK), axes=("AK", "M", "PACK"), order=("AK", "M", "PACK"))
            tl.store(Activation[:, rows, :], activation)
            block_sums = tl.load(x.bsum)
            if not IQ2:
                block_sums = tl.sum_pairs(block_sums)
            sums = tl.reshape(block_sums, (GROUPS, BM), axes=("GROUPS", "M"), order=("K", "M"))
            tl.store(Sums[:, rows], tl.i32(sums))
            tl.store(XScale[rows], tl.load(x.ds))
        with tl.level.cols(N, group=BN) as columns:
            # Author-owned invocation packing has a complete, zero-padded
            # domain. Its layout is consumed directly by the inner byte dots.
            tile = W[columns, block]
            for column in range(BN):
                if column < tl.extent(tile, axis=0):
                    w = tl.load(tile[column])
                    if DECODE is not None:
                        weight, scale, d = DECODE(w, grid, signs)
                        scale = tl.i32(scale)
                    elif AFFINE:
                        weight = tl.u8(w.q[position])
                        if HIGH:
                            weight = weight | tl.u8(w.qh[position]) << tl.u8(4)
                        scale = tl.i32(w.sc[group])
                        d = tl.f32(w.d)
                        minimum = tl.reshape(tl.i32(w.m[group]), (GROUPS,), axes=("GROUPS",), order=("group",))
                        tl.store(Minima[:, column], minimum)
                        tl.store(Dmin[column], tl.f32(w.dmin))
                    elif IQ2:
                        high = (tl.u32(w.qh[code // tl.u32(4)]) >> (code % tl.u32(4) * tl.u32(2))) & tl.u32(3)
                        index = tl.u32(w.q[code]) | (high << tl.u32(8))
                        weight = tl.lookup(grid.values, index * tl.u32(8) + payload, bounds="in_bounds")
                        metadata = tl.u32(w.scales[group // tl.u32(2)])
                        scale = tl.i32((metadata >> (group % tl.u32(2) * tl.u32(4))) & tl.u32(15)) * tl.i32(2) + tl.i32(1)
                        d = tl.f32(w.d) * tl.f32(0.125)
                    elif XXS:
                        weight, scale, position = decode_iq3_xxs(w, grid, signs)
                        scale = tl.i32(scale)
                        d = tl.f32(w.d) * tl.f32(0.25)
                    else:
                        index = tl.u32(w.q[code]) | (tl.u32(w.qh[code]) << tl.u32(8))
                        weight = tl.lookup(grid.values, index * tl.u32(4) + payload, bounds="in_bounds")
                        scale = tl.i32(w.scales[group]) * tl.i32(2) + tl.i32(1)
                        d = tl.f32(w.d)
                    if DECODE is None and not XXS and not AFFINE:
                        sign = tl.i8(1) - tl.i8(w.signs[position]) * tl.i8(2)
                        weight = weight * sign
                    if AFFINE:
                        biased = weight
                    else:
                        biased = tl.u8(weight) + tl.u8(128)
                    panel = tl.reshape(biased, (GROUPS, WORDS, PACK), axes=("GROUPS", "WORDS", "PACK"),
                                       order=("group", "entry", "payload"))
                    tl.store(Panel[column], panel)
                    tl.store(Scales[:, column], tl.reshape(scale, (GROUPS,), axes=("GROUPS",), order=("group",)))
                    tl.store(D[column], d)
                else:
                    zero_scale = tl.i32(tl.arange(0, GROUPS, axis="GROUPS")) * tl.i32(0)
                    zero_word = tl.u8(tl.arange(0, WORDS, axis="WORDS")) * tl.u8(0)
                    zero_byte = tl.u8(tl.arange(0, PACK, axis="PACK")) * tl.u8(0)
                    zero_panel = tl.u8(zero_scale) + zero_word + zero_byte
                    tl.store(Panel[column], zero_panel)
                    tl.store(Scales[:, column], zero_scale)
                    tl.store(D[column], tl.f32(0))
                    if AFFINE:
                        tl.store(Minima[:, column], zero_scale)
                        tl.store(Dmin[column], tl.f32(0))
            d = tl.reshape(tl.load(D), (BN,), axes=("N",), order=("BN",))
            if AFFINE:
                dmin = tl.reshape(tl.load(Dmin), (BN,), axes=("N",), order=("BN",))
            with tl.level.rows(M, group=BM) as rows:
                integer = tl.state(tl.i32, [BN, BM], init=0)
                if AFFINE:
                    minimum = tl.state(tl.i32, [BN, BM], init=0)
                xd = tl.load(XScale[rows])
                for g in range(GROUPS):
                    partial = tl.state(tl.i32, [BN, BM], init=0)
                    for word in tl.static_range(WORDS):
                        panel = tl.reshape(tl.load(Panel[:, g, word]), (BN, 4), axes=("N", "byte"), order=("BN", "PACK"))
                        activation = tl.reshape(tl.load(Activation[g * WORDS + word, rows]), (BM, PACK),
                                                axes=("M", "byte"), order=("M", "PACK"))
                        product = tl.mac_groups(panel, activation, n=4, into=tl.i32)
                        partial += tl.reshape(product, (BN, BM), axes=("N", "M"), order=("N", "M", "byte"))
                    scale = tl.reshape(tl.load(Scales[g]), (BN,), axes=("N",), order=("BN",))
                    sums = tl.load(Sums[g, rows])
                    if AFFINE:
                        integer += partial * scale
                        min_scale = tl.reshape(tl.load(Minima[g]), (BN,), axes=("N",), order=("BN",))
                        minimum += min_scale * sums
                    else:
                        integer += (partial - tl.i32(128) * sums) * scale
                prior = tl.load(Output[columns, rows])
                value = tl.f32(integer) * (tl.f32(xd) * d)
                if AFFINE:
                    value -= tl.f32(minimum) * (tl.f32(xd) * dmin)
                tl.store(Output[columns, rows], prior + value)
    with tl.level.cols(N, group=BN) as columns:
        with tl.level.rows(M, group=BM) as rows:
            value = tl.load(Output[columns, rows])
            value = tl.reshape(value, (BM, BN), axes=("M", "N"), order=("M", "N"))
            tl.store(Y[rows, columns], value)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def iq2_s_tiled(W: tl.View[ggml.IQ2_S, (N, K)], X: tl.View[tl.f32, (M, K)],
                Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X8, (1024, 8)],
                Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 16,
                WORDS: tl.constexpr = 4, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 16 and WORDS == 4 and PACK == 4)
    project_indexed_tiles(W, X, Xq, grid, Y, Panel, Scales, D, Activation, Sums, XScale, Output, N, AK, BM, BN, GROUPS, WORDS, PACK, True)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def iq3_s_tiled(W: tl.View[ggml.IQ3_S, (N, K)], X: tl.View[tl.f32, (M, K)],
                Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X4, (512, 4)],
                Y: tl.View[tl.f32, (M, N)], Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 8,
                WORDS: tl.constexpr = 8, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 8 and WORDS == 8 and PACK == 4)
    project_indexed_tiles(W, X, Xq, grid, Y, Panel, Scales, D, Activation, Sums, XScale, Output, N, AK, BM, BN, GROUPS, WORDS, PACK, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output", "Panel": "panel", "Scales": "scales", "D": "d", "Activation": "activation", "Sums": "sums", "XScale": "xscale", "Output": "accumulation"})
def iq3_xxs_tiled(W: tl.View[ggml.IQ3_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X4, (256, 4)],
                  signs: tl.View[ggml.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)],
                  Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
                  Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
                  Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
                  XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)], AK: tl.constexpr = 64,
                  BM: tl.constexpr = 16, BN: tl.constexpr = 8, GROUPS: tl.constexpr = 8,
                  WORDS: tl.constexpr = 8, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 8 and WORDS == 8 and PACK == 4)
    project_indexed_tiles(W, X, Xq, grid, Y, Panel, Scales, D, Activation, Sums, XScale, Output, N, AK, BM, BN, GROUPS, WORDS, PACK, False, True, signs)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq3_s_prefill(W: tl.View[ggml.IQ3_S, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X4, (512, 4)],
                  Y: tl.View[tl.f32, (M, N)]):
    project_scaled_blocks(W, X, Xq, grid, Y, N, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq3_xxs(W: tl.View[ggml.IQ3_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
            Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X4, (256, 4)],
            Y: tl.View[tl.f32, (M, N)]):
    project(W, X, Xq, grid, None, Y, M, N,
            quant_q8_k.quantize_matrix, compute_iq3_xxs, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq3_xxs_prefill(W: tl.View[ggml.IQ3_XXS, (N, K)], X: tl.View[tl.f32, (M, K)],
                    Xq: tl.View[ggml.Q8_K, (M, K)], grid: tl.View[ggml.I8X4, (256, 4)],
                    signs: tl.View[ggml.I8X8, (128, 8)], Y: tl.View[tl.f32, (M, N)]):
    project_scaled_blocks(W, X, Xq, grid, Y, N, False, True, signs)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq4_nl_specialized(W: tl.View[ggml.IQ4_NL, (N, K)], X: tl.View[tl.f32, (M, K)],
                        Xq: tl.View[ggml.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
                        Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    project(W, X, Xq, codebook, None, Y, M, N, quant_q8_0.quantize_matrix, compute_iq4_nl, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def iq4_xs_specialized(W: tl.View[ggml.IQ4_XS, (N, K)], X: tl.View[tl.f32, (M, K)],
                        Xq: tl.View[ggml.Q8_K, (M, K)], codebook: tl.View[tl.i8, (16,)],
                        Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    project(W, X, Xq, codebook, None, Y, M, N, quant_q8_k.quantize_matrix, compute_iq4_xs, False)


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def mxfp4_specialized(W: tl.View[ggml.MXFP4, (N, K)], X: tl.View[tl.f32, (M, K)],
                        Xq: tl.View[ggml.Q8_0, (M, K)], codebook: tl.View[tl.i8, (16,)],
                        scale: tl.View[tl.f32, (256,)], Y: tl.View[tl.f32, (M, N)], *, K: tl.constexpr):
    project(W, X, Xq, codebook, scale, Y, M, N, quant_q8_0.quantize_matrix, compute_mxfp4, True)



@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Y": "output", "Panel": "panel",
    "Scales": "scales", "D": "scale", "Activation": "activation", "Sums": "sums",
    "XScale": "xscale", "Output": "accumulation", "Minima": "minima", "Dmin": "minimum_scale"})
def q4_k_tiled(W: tl.View[ggml.Q4_K, (N, K)], X: tl.View[tl.f32, (M, K)],
               Xq: tl.View[ggml.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
               Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
               Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
               Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
               XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)],
               Minima: tl.View[tl.i32, (GROUPS, BN)], Dmin: tl.View[tl.f32, (BN,)],
               AK: tl.constexpr = 64, BM: tl.constexpr = 16, BN: tl.constexpr = 8,
               GROUPS: tl.constexpr = 8, WORDS: tl.constexpr = 8, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 8 and WORDS == 8 and PACK == 4)
    project_indexed_tiles(W, X, Xq, None, Y, Panel, Scales, D, Activation, Sums, XScale,
                          Output, N, AK, BM, BN, GROUPS, WORDS, PACK, False, False, None,
                          True, False, Minima, Dmin)


@triton.jit(model="weft", alias_groups={"Xq": "quantized", "Y": "output", "Panel": "panel",
    "Scales": "scales", "D": "scale", "Activation": "activation", "Sums": "sums",
    "XScale": "xscale", "Output": "accumulation", "Minima": "minima", "Dmin": "minimum_scale"})
def q5_k_tiled(W: tl.View[ggml.Q5_K, (N, K)], X: tl.View[tl.f32, (M, K)],
               Xq: tl.View[ggml.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
               Panel: tl.View[tl.u8, (BN, GROUPS, WORDS, PACK)],
               Scales: tl.View[tl.i32, (GROUPS, BN)], D: tl.View[tl.f32, (BN,)],
               Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
               XScale: tl.View[tl.f32, (M,)], Output: tl.View[tl.f32, (N, M)],
               Minima: tl.View[tl.i32, (GROUPS, BN)], Dmin: tl.View[tl.f32, (BN,)],
               AK: tl.constexpr = 64, BM: tl.constexpr = 16, BN: tl.constexpr = 8,
               GROUPS: tl.constexpr = 8, WORDS: tl.constexpr = 8, PACK: tl.constexpr = 4):
    tl.static_assert(GROUPS == 8 and WORDS == 8 and PACK == 4)
    project_indexed_tiles(W, X, Xq, None, Y, Panel, Scales, D, Activation, Sums, XScale,
                          Output, N, AK, BM, BN, GROUPS, WORDS, PACK, False, False, None,
                          True, True, Minima, Dmin)
