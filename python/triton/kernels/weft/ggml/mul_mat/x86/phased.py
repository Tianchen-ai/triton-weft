"""Invocation-owned shared preparation, followed by independent output tiles."""
from __future__ import annotations
import triton
import triton.language as tl
from triton.integrations import ggml_formats
from ...quantize.q8_k_first import q8_k_quantize_first_absolute
from .codebook import indexed_tile_product


@triton.jit
def activation_fields(Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Activation: tl.View[tl.i8, (AK, M, PACK)], Sums: tl.View[tl.i32, (GROUPS, M)],
    XScale: tl.View[tl.f32, (M,)], *, BM: tl.constexpr, AK: tl.constexpr,
    PACK: tl.constexpr, GROUPS: tl.constexpr):
    # Transpose one encoded record at a time instead of constructing a large
    # row-by-record register value before writing the invocation-owned layout.
    for row in range(M):
        values = tl.load(Xq[row])
        packed = tl.reshape(values.q, (AK, PACK), axes=("AK", "PACK"), order=("K",))
        tl.store(Activation[:, row, :], packed)
        bsums = tl.reshape(tl.sum_pairs(values.bsum), (GROUPS,), axes=("GROUPS",), order=("K",))
        tl.store(Sums[:, row], tl.i32(bsums))
        tl.store(XScale[row], values.ds)


@triton.jit(model="weft", alias_groups={"X": "input", "Xq": "quantized", "Activation": "activation",
    "Sums": "sums", "XScale": "xscale"})
def q4_prepare_activation(X: tl.View[tl.f32, (M, K)], Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    Activation: tl.View[tl.i8, (B, AK, M, PACK)], Sums: tl.View[tl.i32, (B, GROUPS, M)],
    XScale: tl.View[tl.f32, (B, M)], *, M: tl.constexpr, K: tl.constexpr, B: tl.constexpr,
    PARTS: tl.constexpr, GRAIN: tl.constexpr, BM: tl.constexpr = 16,
    AK: tl.constexpr = 64, PACK: tl.constexpr = 4, GROUPS: tl.constexpr = 8):
    part = tl.program_id(0)
    tl.assume(part >= 0)
    tl.assume(part < PARTS)
    first = part * GRAIN
    x = tl.subview(X, offsets=(first, 0), extents=(GRAIN, K))
    xq = tl.subview(Xq, offsets=(first, 0), extents=(GRAIN, K))
    for row in range(GRAIN):
        q8_k_quantize_first_absolute(x[row], xq[row])
    for b in range(B):
        block = tl.subview(xq, offsets=(0, b * 256), extents=(GRAIN, 256))
        activation = tl.subview(Activation[b], offsets=(0, first, 0), extents=(AK, GRAIN, PACK))
        sums = tl.subview(Sums[b], offsets=(0, first), extents=(GROUPS, GRAIN))
        scale = tl.subview(XScale[b], offsets=(first,), extents=(GRAIN,))
        activation_fields(block, activation, sums, scale, BM=BM, AK=AK, PACK=PACK, GROUPS=GROUPS)


@triton.jit(model="weft", alias_groups={"X": "input", "Xq": "quantized"})
def prepare_quantized_activation(X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)], *, M: tl.constexpr, K: tl.constexpr,
    PARTS: tl.constexpr, GRAIN: tl.constexpr):
    part = tl.program_id(0)
    tl.assume(part >= 0)
    tl.assume(part < PARTS)
    for row in range(GRAIN):
        position = part * GRAIN + row
        q8_k_quantize_first_absolute(X[position], Xq[position])


@triton.jit(model="weft", alias_groups={"W": "weights", "Panel": "panel", "Scales": "scales",
    "D": "scale", "Minima": "minima", "Dmin": "minimum_scale"})
def q4_prepare_weights(W: tl.View[ggml_formats.Q4_K, (N, K)],
    Panel: tl.View[tl.u8, (B, TILES, BN, GROUPS, WORDS, PACK)],
    Scales: tl.View[tl.i32, (B, TILES, GROUPS, BN)], D: tl.View[tl.f32, (B, TILES, BN)],
    Minima: tl.View[tl.i32, (B, TILES, GROUPS, BN)], Dmin: tl.View[tl.f32, (B, TILES, BN)],
    *, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr, TILES: tl.constexpr,
    BN: tl.constexpr = 8, GROUPS: tl.constexpr = 8, WORDS: tl.constexpr = 8, PACK: tl.constexpr = 4):
    tile = tl.program_id(0)
    tl.assume(tile >= 0)
    tl.assume(tile < TILES)
    group = tl.arange(0, GROUPS, dtype=tl.u32, axis="group")
    word = tl.arange(0, WORDS, dtype=tl.u32, axis="entry")
    byte = tl.arange(0, PACK, dtype=tl.u32, axis="payload")
    position = group * tl.u32(32) + word * tl.u32(4) + byte
    for b in range(B):
        for column in range(BN):
            block = tl.subview(W[tile * BN + column], offsets=(b * 256,), extents=(256,))
            w = tl.load(block)
            packed = tl.reshape(tl.u8(w.q[position]), (GROUPS, WORDS, PACK),
                axes=("GROUPS", "WORDS", "PACK"), order=("group", "entry", "payload"))
            tl.store(Panel[b, tile, column], packed)
            tl.store(Scales[b, tile, :, column], tl.reshape(tl.i32(w.sc[group]),
                (GROUPS,), axes=("GROUPS",), order=("group",)))
            tl.store(Minima[b, tile, :, column], tl.reshape(tl.i32(w.m[group]),
                (GROUPS,), axes=("GROUPS",), order=("group",)))
            tl.store(D[b, tile, column], tl.f32(w.d))
            tl.store(Dmin[b, tile, column], tl.f32(w.dmin))


@triton.jit
def consume_output_tile(Y: tl.View[tl.f32, (M, N)], Panel, Scales, D, Activation, Sums,
    XScale, Minima, Dmin, tile, *, B: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    GROUPS: tl.constexpr, WORDS: tl.constexpr, PACK: tl.constexpr):
    with tl.level.cols(N, group=BN) as columns:
        with tl.level.rows(M, group=BM) as rows:
            result = tl.state(tl.f32, [BN, BM], init=0)
            for b in range(B):
                result += indexed_tile_product(Panel[b, tile], Scales[b, tile], D[b, tile],
                    Activation[b], Sums[b], XScale[b], rows, columns, BM, BN, GROUPS, WORDS, PACK,
                    True, Minima[b, tile], Dmin[b, tile])
            transposed = tl.reshape(result, (BM, BN), axes=("M", "N"), order=("M", "N"))
            tl.store(Y[rows, columns], transposed)


@triton.jit(model="weft", alias_groups={"Y": "output", "Panel": "panel", "Scales": "scales",
    "D": "scale", "Activation": "activation", "Sums": "sums", "XScale": "xscale",
    "Minima": "minima", "Dmin": "minimum_scale"})
def q4_compute(Y: tl.View[tl.f32, (M, N)],
    Panel: tl.View[tl.u8, (B, TILES, BN, GROUPS, WORDS, PACK)],
    Scales: tl.View[tl.i32, (B, TILES, GROUPS, BN)], D: tl.View[tl.f32, (B, TILES, BN)],
    Activation: tl.View[tl.i8, (B, AK, M, PACK)], Sums: tl.View[tl.i32, (B, GROUPS, M)],
    XScale: tl.View[tl.f32, (B, M)], Minima: tl.View[tl.i32, (B, TILES, GROUPS, BN)],
    Dmin: tl.View[tl.f32, (B, TILES, BN)], *, M: tl.constexpr, N: tl.constexpr,
    B: tl.constexpr, TILES: tl.constexpr, BM: tl.constexpr = 16, BN: tl.constexpr = 8,
    AK: tl.constexpr = 64, GROUPS: tl.constexpr = 8, WORDS: tl.constexpr = 8, PACK: tl.constexpr = 4):
    tile = tl.program_id(0)
    tl.assume(tile >= 0)
    tl.assume(tile < TILES)
    output = tl.subview(Y, offsets=(0, tile * BN), extents=(M, BN))
    consume_output_tile(output, Panel, Scales, D, Activation, Sums, XScale, Minima, Dmin,
        tile, B=B, BM=BM, BN=BN, GROUPS=GROUPS, WORDS=WORDS, PACK=PACK)


@triton.jit(model="weft", alias_groups={"W": "weights", "Y": "output", "Panel": "panel",
    "Scales": "scales", "D": "scale", "Activation": "activation", "Sums": "sums",
    "XScale": "xscale", "Minima": "minima", "Dmin": "minimum_scale"})
def q4_prepare_and_compute(W: tl.View[ggml_formats.Q4_K, (N, K)], Y: tl.View[tl.f32, (M, N)],
    Panel: tl.View[tl.u8, (B, TILES, BN, GROUPS, WORDS, PACK)],
    Scales: tl.View[tl.i32, (B, TILES, GROUPS, BN)], D: tl.View[tl.f32, (B, TILES, BN)],
    Activation: tl.View[tl.i8, (B, AK, M, PACK)], Sums: tl.View[tl.i32, (B, GROUPS, M)],
    XScale: tl.View[tl.f32, (B, M)], Minima: tl.View[tl.i32, (B, TILES, GROUPS, BN)],
    Dmin: tl.View[tl.f32, (B, TILES, BN)], *, M: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, B: tl.constexpr, TILES: tl.constexpr, BM: tl.constexpr = 16,
    BN: tl.constexpr = 8, AK: tl.constexpr = 64, GROUPS: tl.constexpr = 8,
    WORDS: tl.constexpr = 8, PACK: tl.constexpr = 4):
    # Each program owns one output-column tile and its weight preparation;
    # only the shared activation preparation needs a cross-program barrier.
    q4_prepare_weights(W, Panel, Scales, D, Minima, Dmin, N=N, K=K, B=B,
        TILES=TILES, BN=BN, GROUPS=GROUPS, WORDS=WORDS, PACK=PACK)
    q4_compute(Y, Panel, Scales, D, Activation, Sums, XScale, Minima, Dmin,
        M=M, N=N, B=B, TILES=TILES, BM=BM, BN=BN, AK=AK,
        GROUPS=GROUPS, WORDS=WORDS, PACK=PACK)
