from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as quantization
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"W": "weights", "Y": "output"})
def q8_0(W: tl.View[ggml.Q8_0, (K,)], Y: tl.View[tl.f32, (K,)]):
    """Decode disjoint rows, with the same non-alias contract as GGML."""
    with tl.level.blocks(K, extent=32) as block:
        record = tl.load(W[block])
        tl.store(Y[block], quantization.symmetric_integer(record.q, record.d))


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q2_k(W: tl.View[ggml.Q2_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=256) as block:
        record = tl.load(W[block])
        group = tl.arange(0, 16, dtype=tl.u32, axis="group")
        payload = tl.arange(0, 16, dtype=tl.u32, axis="payload")
        metadata = tl.u32(record.scales[group])
        value = quantization.k_superblock(record.q[group * tl.u32(16) + payload], metadata & tl.u32(15),
                                          record.d, metadata >> tl.u32(4), record.dmin)
        tl.store(Y[block], tl.reshape(value, (256,), axes=("k",), order=("group", "payload")))


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q3_k(W: tl.View[ggml.Q3_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=256) as block:
        record = tl.load(W[block])
        lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
        group = lane // tl.u32(16)
        scale = (tl.i32(record.scale_low[group]) | tl.i32(record.scale_high[group]) << tl.u32(4)) - tl.i32(32)
        q = tl.i32(record.q[lane]) + tl.i32(record.hmask[lane]) * tl.i32(4) - tl.i32(4)
        tl.store(Y[block], tl.f32(record.d) * tl.f32(scale) * tl.f32(q))


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q5_k(W: tl.View[ggml.Q5_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    with tl.level.blocks(K, extent=256) as block:
        record = tl.load(W[block])
        lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
        group = lane // tl.u32(32)
        q = tl.i32(record.q[lane]) | tl.i32(record.qh[lane]) << tl.u32(4)
        value = quantization.k_superblock(q, record.sc[group], record.d, record.m[group], record.dmin)
        tl.store(Y[block], value)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_k(W: tl.View[ggml.Q4_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    """Decode one independent record as a Value, preserving GGML scale order."""
    with tl.level.blocks(K, extent=256) as block:
        record = tl.load(W[block])
        lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
        group = lane // tl.u32(32)
        value = quantization.k_superblock(
            record.q[lane], record.sc[group], record.d, record.m[group], record.dmin
        )
        tl.store(Y[block], value)


@triton.jit(model="weft")
def q6_k(W: tl.View[ggml.Q6_K, (K,)], Y: tl.View[tl.f32, (K,)]):
    """Expose independent bitplanes and scales without a dynamic local gather."""
    with tl.level.blocks(K, extent=256) as block:
        record = tl.load(W[block])
        lane = tl.arange(0, 256, dtype=tl.u32, axis="k")
        q = tl.i32(record.ql[lane]) | tl.i32(record.qh[lane]) << tl.u32(4)
        q -= tl.i32(32)
        value = tl.f32(record.d) * tl.f32(record.scales[lane // tl.u32(16)]) * tl.f32(q)
        tl.store(Y[block], value)


@triton.jit
def store_entries(Y, value, ENTRIES: tl.constexpr, PAYLOAD: tl.constexpr):
    flat = tl.reshape(value, (ENTRIES * PAYLOAD,), axes=("k",), order=("entry", "payload"))
    tl.store(Y, flat)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq1_s(W: tl.View[ggml.IQ1_S, (K,)], grid: tl.View[tl.i8, (16384,)],
          Y: tl.View[tl.f32, (K,)]):
    entry = tl.arange(0, 32, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        metadata = tl.u32(w.qh[entry // tl.u32(4)])
        high = (metadata >> (entry % tl.u32(4) * tl.u32(3))) & tl.u32(7)
        index = tl.u32(w.q[entry]) | (high << tl.u32(8))
        value = tl.lookup(grid, index * tl.u32(8) + payload, bounds="in_bounds")
        delta = tl.where((metadata & tl.u32(32768)) != tl.u32(0), tl.f32(-0.125), tl.f32(0.125))
        scale = tl.f32(w.d) * tl.f32(((metadata >> tl.u32(12)) & tl.u32(7)) * tl.u32(2) + tl.u32(1))
        store_entries(Y[block], scale * (tl.f32(value) + delta), 32, 8)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq1_m(W: tl.View[ggml.IQ1_M, (K,)], grid: tl.View[tl.i8, (16384,)],
          f16_bits: tl.View[tl.f32, (65536,)], Y: tl.View[tl.f32, (K,)]):
    entry = tl.arange(0, 32, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        sc0 = tl.u32(w.scales[0]) | tl.u32(w.scales[1]) << tl.u32(8)
        sc1 = tl.u32(w.scales[2]) | tl.u32(w.scales[3]) << tl.u32(8)
        sc2 = tl.u32(w.scales[4]) | tl.u32(w.scales[5]) << tl.u32(8)
        sc3 = tl.u32(w.scales[6]) | tl.u32(w.scales[7]) << tl.u32(8)
        bits = (sc0 >> tl.u32(12)) | ((sc1 >> tl.u32(8)) & tl.u32(240))
        bits = bits | ((sc2 >> tl.u32(4)) & tl.u32(3840)) | (sc3 & tl.u32(61440))
        d = tl.lookup(f16_bits, bits, bounds="in_bounds")
        metadata = tl.u32(w.qh[entry // tl.u32(2)]) >> (entry % tl.u32(2) * tl.u32(4))
        index = tl.u32(w.q[entry]) | ((metadata & tl.u32(7)) << tl.u32(8))
        value = tl.lookup(grid, index * tl.u32(8) + payload, bounds="in_bounds")
        delta = tl.where((metadata & tl.u32(8)) != tl.u32(0), tl.f32(-0.125), tl.f32(0.125))
        group = entry // tl.u32(4)
        sc = tl.u32(w.scales[group // tl.u32(2) * tl.u32(2)])
        sc = sc | tl.u32(w.scales[group // tl.u32(2) * tl.u32(2) + tl.u32(1)]) << tl.u32(8)
        shift = group % tl.u32(2) * tl.u32(6) + entry % tl.u32(4) // tl.u32(2) * tl.u32(3)
        scale = d * tl.f32(((sc >> shift) & tl.u32(7)) * tl.u32(2) + tl.u32(1))
        store_entries(Y[block], scale * (tl.f32(value) + delta), 32, 8)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq2_s(W: tl.View[ggml.IQ2_S, (K,)], grid: tl.View[tl.i8, (8192,)],
          Y: tl.View[tl.f32, (K,)]):
    entry = tl.arange(0, 32, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        high = (tl.u32(w.qh[entry // tl.u32(4)]) >> (entry % tl.u32(4) * tl.u32(2))) & tl.u32(3)
        index = tl.u32(w.q[entry]) | (high << tl.u32(8))
        value = tl.lookup(grid, index * tl.u32(8) + payload, bounds="in_bounds")
        sign = tl.i32(1) - tl.i32(w.signs[entry * tl.u32(8) + payload]) * tl.i32(2)
        scale = (tl.u32(w.scales[entry // tl.u32(4)]) >> (entry % tl.u32(4) // tl.u32(2) * tl.u32(4))) & tl.u32(15)
        d = tl.f32(w.d) * (tl.f32(0.5) + tl.f32(scale)) * tl.f32(0.25)
        store_entries(Y[block], d * tl.f32(tl.i32(value) * sign), 32, 8)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq2_xs(W: tl.View[ggml.IQ2_XS, (K,)], grid: tl.View[tl.i8, (4096,)],
           signs: tl.View[tl.i8, (1024,)], Y: tl.View[tl.f32, (K,)]):
    entry = tl.arange(0, 32, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        word = tl.u32(w.q[entry])
        value = tl.lookup(grid, (word & tl.u32(511)) * tl.u32(8) + payload, bounds="in_bounds")
        sign = tl.lookup(signs, (word >> tl.u32(9)) * tl.u32(8) + payload, bounds="in_bounds")
        scale = (tl.u32(w.scales[entry // tl.u32(4)]) >> (entry % tl.u32(4) // tl.u32(2) * tl.u32(4))) & tl.u32(15)
        d = tl.f32(w.d) * (tl.f32(0.5) + tl.f32(scale)) * tl.f32(0.25)
        store_entries(Y[block], d * tl.f32(tl.i32(value) * tl.i32(sign)), 32, 8)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq2_xxs(W: tl.View[ggml.IQ2_XXS, (K,)], grid: tl.View[tl.i8, (2048,)],
            signs: tl.View[tl.i8, (1024,)], Y: tl.View[tl.f32, (K,)]):
    entry = tl.arange(0, 32, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        group = entry // tl.u32(4)
        metadata = tl.u32(w.q[group * tl.u32(4) + tl.u32(2)]) | tl.u32(w.q[group * tl.u32(4) + tl.u32(3)]) << tl.u32(16)
        word = tl.u32(w.q[group * tl.u32(4) + entry % tl.u32(4) // tl.u32(2)])
        index = (word >> (entry % tl.u32(2) * tl.u32(8))) & tl.u32(255)
        sign_index = (metadata >> (entry % tl.u32(4) * tl.u32(7))) & tl.u32(127)
        value = tl.lookup(grid, index * tl.u32(8) + payload, bounds="in_bounds")
        sign = tl.lookup(signs, sign_index * tl.u32(8) + payload, bounds="in_bounds")
        d = tl.f32(w.d) * (tl.f32(0.5) + tl.f32(metadata >> tl.u32(28))) * tl.f32(0.25)
        store_entries(Y[block], d * tl.f32(tl.i32(value) * tl.i32(sign)), 32, 8)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq3_s(W: tl.View[ggml.IQ3_S, (K,)], grid: tl.View[tl.i8, (2048,)],
          Y: tl.View[tl.f32, (K,)]):
    entry = tl.arange(0, 64, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 4, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        index = tl.u32(w.q[entry]) | tl.u32(w.qh[entry]) << tl.u32(8)
        value = tl.lookup(grid, index * tl.u32(4) + payload, bounds="in_bounds")
        sign = tl.i32(1) - tl.i32(w.signs[entry * tl.u32(4) + payload]) * tl.i32(2)
        d = tl.f32(w.d) * tl.f32(tl.i32(w.scales[entry // tl.u32(8)]) * tl.i32(2) + tl.i32(1))
        store_entries(Y[block], d * tl.f32(tl.i32(value) * sign), 64, 4)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def iq3_xxs(W: tl.View[ggml.IQ3_XXS, (K,)], grid: tl.View[tl.i8, (1024,)],
            signs: tl.View[tl.i8, (1024,)], Y: tl.View[tl.f32, (K,)]):
    entry = tl.arange(0, 64, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 4, dtype=tl.u32, axis="payload")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        metadata = tl.u32(w.metadata[entry // tl.u32(8)])
        value = tl.lookup(grid, tl.u32(w.q[entry]) * tl.u32(4) + payload, bounds="in_bounds")
        sign_index = (metadata >> (entry % tl.u32(8) // tl.u32(2) * tl.u32(7))) & tl.u32(127)
        sign = tl.lookup(signs, sign_index * tl.u32(8) + entry % tl.u32(2) * tl.u32(4) + payload, bounds="in_bounds")
        d = tl.f32(w.d) * (tl.f32(0.5) + tl.f32(metadata >> tl.u32(28))) * tl.f32(0.5)
        store_entries(Y[block], d * tl.f32(tl.i32(value) * tl.i32(sign)), 64, 4)
