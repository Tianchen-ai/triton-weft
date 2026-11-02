from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit
def decode_records(W, Y, K, RECORDS: tl.constexpr, CENTER: tl.constexpr,
                   HIGH: tl.constexpr, AFFINE: tl.constexpr, CACHE: tl.constexpr = ""):
    with tl.level.blocks(K, extent=32 * RECORDS) as block:
        w = tl.reshape(tl.load(W[block]), (RECORDS, 32), axes=("records", "K"), order=("K",))
        q = tl.i32(w.q)
        if HIGH:
            q = q | (tl.i32(w.qh) << 4)
        q = tl.reshape(q, (RECORDS, 32), axes=("records", "lane"), order=("records", "K"))
        scale = tl.f32(w.d)
        value = tl.f32(q - CENTER) * scale
        if AFFINE:
            value += tl.f32(w.m)
        tl.store(Y[block], tl.reshape(value, (RECORDS * 32,), axes=("K",), order=("records", "lane")), cache_modifier=CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_0_records(W: tl.View[ggml_formats.Q4_0, (K,)], Y: tl.View[tl.f32, (K,)], RECORDS: tl.constexpr = 4):
    decode_records(W, Y, K, RECORDS, 8, False, False)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_1_records(W: tl.View[ggml_formats.Q4_1, (K,)], Y: tl.View[tl.f32, (K,)], RECORDS: tl.constexpr = 4):
    decode_records(W, Y, K, RECORDS, 0, False, True)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q5_0_records(W: tl.View[ggml_formats.Q5_0, (K,)], Y: tl.View[tl.f32, (K,)], RECORDS: tl.constexpr = 4):
    decode_records(W, Y, K, RECORDS, 16, True, False)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q5_1_records(W: tl.View[ggml_formats.Q5_1, (K,)], Y: tl.View[tl.f32, (K,)], RECORDS: tl.constexpr = 4):
    decode_records(W, Y, K, RECORDS, 0, True, True)


@triton.jit
def decode_codebook_records(W, codebook, scales, Y, K,
                            RECORDS: tl.constexpr, EXPONENT: tl.constexpr, CACHE: tl.constexpr = ""):
    table = tl.stage(tl.load(codebook))
    with tl.level.blocks(K, extent=32 * RECORDS) as block:
        w = tl.reshape(tl.load(W[block]), (RECORDS, 32), axes=("records", "K"), order=("K",))
        q = tl.lookup(table, w.q, bounds="in_bounds")
        q = tl.reshape(q, (RECORDS, 32), axes=("records", "lane"), order=("records", "K"))
        if EXPONENT:
            d = tl.lookup(scales, w.e, bounds="in_bounds")
        else:
            d = tl.f32(w.d)
        value = tl.f32(q) * d
        tl.store(Y[block], tl.reshape(value, (RECORDS * 32,), axes=("K",), order=("records", "lane")), cache_modifier=CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq4_nl_records(W: tl.View[ggml_formats.IQ4_NL, (K,)], codebook: tl.View[tl.i8, (16,)],
                   Y: tl.View[tl.f32, (K,)], RECORDS: tl.constexpr = 2):
    decode_codebook_records(W, codebook, None, Y, K, RECORDS, False)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def mxfp4_records(W: tl.View[ggml_formats.MXFP4, (K,)], codebook: tl.View[tl.i8, (16,)],
                  scale: tl.View[tl.f32, (256,)], Y: tl.View[tl.f32, (K,)], RECORDS: tl.constexpr = 2):
    decode_codebook_records(W, codebook, scale, Y, K, RECORDS, True)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q8_0(W: tl.View[ggml_formats.Q8_0, (K,)], Y: tl.View[tl.f32, (K,)], LANES: tl.constexpr = 8,
         CACHE: tl.constexpr = ""):
    tl.static_assert(LANES == 4 or LANES == 8 or LANES == 16)
    with tl.level.blocks(K, extent=32) as block:
        if tl.extent(W[block], axis=0) >= 32:
            w = W[block]
            scale = tl.f32(tl.load(w.d))
            for part in tl.static_range(32 // LANES):
                values = tl.load(tl.subview(w.q, offsets=(part * LANES,), extents=(LANES,)))
                value = scale * tl.f32(values)
                tl.store(tl.subview(Y[block], offsets=(part * LANES,), extents=(LANES,)), value, cache_modifier=CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (K,)], grid: tl.View[tl.i8, (16384,)],
          Y: tl.View[tl.f32, (K,)], ENTRIES: tl.constexpr = 8, CACHE: tl.constexpr = ""):
    """Decode bounded entry tiles from one immutable record snapshot."""
    tl.static_assert(ENTRIES == 4 or ENTRIES == 8 or ENTRIES == 16)
    entry = tl.arange(0, ENTRIES, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        if tl.extent(W[block], axis=0) >= 256:
            w = tl.load(W[block])
            for part in tl.static_range(32 // ENTRIES):
                index = entry + tl.u32(part * ENTRIES)
                metadata = tl.u16(w.qh[index // tl.u32(4)])
                high = (metadata >> tl.u16(index % tl.u32(4) * tl.u32(3))) & tl.u16(7)
                code = tl.u16(w.q[index]) | (high << tl.u16(8))
                value = tl.lookup(grid, code * tl.u16(8) + tl.u16(payload), bounds="in_bounds")
                delta = tl.where((metadata & tl.u16(32768)) != tl.u16(0), tl.f32(-0.125), tl.f32(0.125))
                scale = tl.f32(w.d) * tl.f32(((metadata >> tl.u16(12)) & tl.u16(7)) * tl.u16(2) + tl.u16(1))
                decoded = tl.reshape(scale * (tl.f32(value) + delta), (ENTRIES * 8,),
                                     axes=("k",), order=("entry", "payload"))
                tl.store(tl.subview(Y[block], offsets=(part * ENTRIES * 8,), extents=(ENTRIES * 8,)), decoded, cache_modifier=CACHE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def tq2_0(W: tl.View[ggml_formats.TQ2_0, (K,)], Y: tl.View[tl.f32, (K,)], LANES: tl.constexpr = 32):
    tl.static_assert(LANES == 16 or LANES == 32 or LANES == 64)
    lane = tl.arange(0, LANES, dtype=tl.u16, axis="k")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        for part in tl.static_range(256 // LANES):
            value = tl.f32(tl.i8(w.q[lane + tl.u16(part * LANES)]) - tl.i8(1)) * tl.f32(w.d)
            tl.store(tl.subview(Y[block], offsets=(part * LANES,), extents=(LANES,)), value)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q2_k(W: tl.View[ggml_formats.Q2_K, (K,)], Y: tl.View[tl.f32, (K,)], GROUPS: tl.constexpr = 2,
         CACHE: tl.constexpr = ""):
    tl.static_assert(GROUPS == 2 or GROUPS == 4 or GROUPS == 8)
    group = tl.arange(0, GROUPS, dtype=tl.u32, axis="group")
    payload = tl.arange(0, 16, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        if tl.extent(W[block], axis=0) >= 256:
            w = tl.load(W[block])
            for part in tl.static_range(16 // GROUPS):
                index = group + tl.u32(part * GROUPS)
                metadata = tl.u8(w.scales[index])
                scale = tl.f32(w.d) * tl.f32(metadata & tl.u8(15))
                minimum = tl.f32(w.dmin) * tl.f32(metadata >> tl.u8(4))
                value = scale * tl.f32(w.q[index * tl.u32(16) + payload]) - minimum
                decoded = tl.reshape(value, (GROUPS * 16,), axes=("k",), order=("group", "payload"))
                tl.store(tl.subview(Y[block], offsets=(part * GROUPS * 16,), extents=(GROUPS * 16,)), decoded, cache_modifier=CACHE)
