from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from .signs import even_parity_signs

PARTIALS = tl.constexpr(16)


@triton.jit
def compute_paired(W, X: tl.View[ggml_formats.Q8_K, (K,)], grid,
                   IQ2: tl.constexpr):
    """Keep scaled integer groups across blocks before one float sum."""
    acc = tl.state(tl.f32, [PARTIALS], init=0)
    table = grid.values
    if IQ2:
        entry = tl.arange(0, 8, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 8, dtype=tl.u32, axis="payload")
    else:
        entry = tl.arange(0, 16, dtype=tl.u32, axis="entry")
        payload = tl.arange(0, 4, dtype=tl.u32, axis="payload")
    if IQ2:
        pair = tl.arange(0, 32, dtype=tl.u32, axis="k")
    else:
        pair = tl.arange(0, 16, dtype=tl.u32, axis="k")
    correction_group = tl.arange(0, 16, dtype=tl.u32, axis="PARTIALS")
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        x = tl.load(X[block])
        integer = tl.state(tl.i32, [PARTIALS], init=0)
        for part in tl.static_range(4):
            if IQ2:
                code = tl.u32(part * 8) + entry
                high = (tl.u32(w.qh[code // tl.u32(4)]) >>
                        (code % tl.u32(4) * tl.u32(2))) & tl.u32(3)
                index = tl.u32(w.q[code]) | (high << tl.u32(8))
                position = tl.u32(part * 64) + entry * tl.u32(8) + payload
                weight = tl.lookup(table, index * tl.u32(8) + payload, bounds="in_bounds")
                group = tl.u32(part * 4) + pair // tl.u32(8)
                metadata = tl.u32(w.scales[group // tl.u32(2)])
                scale = tl.i16((metadata >> (group % tl.u32(2) * tl.u32(4))) & tl.u32(15)) * tl.i16(2) + tl.i16(1)
            else:
                code = tl.u32(part * 16) + entry
                index = tl.u32(w.q[code]) | (tl.u32(w.qh[code]) << tl.u32(8))
                position = tl.u32(part * 64) + entry * tl.u32(4) + payload
                weight = tl.lookup(table, index * tl.u32(4) + payload, bounds="in_bounds")
                scale = tl.i16(w.scales[tl.u32(part * 2) + pair // tl.u32(8)]) * tl.i16(2) + tl.i16(1)
            if IQ2:
                # Positive grid entries fit U6; Q8_K's [-127,127] domain
                # permits exact sign transfer and non-saturating I16 pairs.
                signed_activation = tl.where(w.signs[position] != 0, -x.q[position], x.q[position])
                weights = tl.reshape(tl.u6(weight), (64,), axes=("k",), order=("entry", "payload"))
                activation = tl.reshape(signed_activation, (64,), axes=("k",), order=("entry", "payload"))
                products = tl.mac_pairs(weights, activation, into=tl.i16)
                scaled = tl.mac_pairs(products, scale, into=tl.i32)
            else:
                signed = tl.where(w.signs[position] != 0, -weight, weight)
                weights = tl.reshape(tl.u8(signed) + tl.u8(128), (64,), axes=("k",), order=("entry", "payload"))
                activation = tl.reshape(x.q[position], (64,), axes=("k",), order=("entry", "payload"))
                products = tl.mac_groups(weights, activation, n=4, into=tl.i32)
                scaled = products * tl.i32(scale)
            integer += tl.reshape(scaled, (PARTIALS,), axes=("PARTIALS",), order=("k",))
        if not IQ2:
            correction_scale = tl.i32(w.scales[correction_group // tl.u32(2)]) * tl.i32(2) + tl.i32(1)
            integer -= tl.i32(128) * tl.i32(x.bsum[correction_group]) * correction_scale
        acc += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(integer)
    if IQ2:
        return tl.f32(0.125) * tl.sum(acc, 0)
    return tl.sum(acc, 0)


@triton.jit
def compute_iq2_s(W, X: tl.View[ggml_formats.Q8_K, (K,)], grid):
    return compute_paired(W, X, grid, True)


@triton.jit
def compute_iq3_s(W, X: tl.View[ggml_formats.Q8_K, (K,)], grid):
    return compute_paired(W, X, grid, False)


@triton.jit
def decode_iq3_xxs(w, grid, signs, PARITY: tl.constexpr = False):
    """Expose codebook entries and sign rows as complete logical axes."""
    group = tl.arange(0, 8, dtype=tl.u32, axis="group")
    entry = tl.arange(0, 8, dtype=tl.u32, axis="entry")
    payload = tl.arange(0, 4, dtype=tl.u32, axis="payload")
    sign_entry = tl.arange(0, 4, dtype=tl.u32, axis="sign_entry")
    sign_payload = tl.arange(0, 8, dtype=tl.u32, axis="sign_payload")
    metadata = w.metadata[group]
    index = tl.u32(w.q[group * tl.u32(8) + entry])
    weight = tl.lookup(grid.values, index * tl.u32(4) + payload, bounds="in_bounds")
    sign_index = (metadata >> (sign_entry * tl.u32(7))) & tl.u32(127)
    if PARITY:
        sign = even_parity_signs(sign_index, sign_payload)
    else:
        sign = tl.lookup(signs.values, sign_index * tl.u32(8) + sign_payload, bounds="in_bounds")
    sign = tl.reshape(sign, (8, 8, 4), axes=("group", "entry", "payload"),
                      order=("group", "sign_entry", "sign_payload"))
    scale = tl.i16(metadata >> tl.u32(28)) * tl.i16(2) + tl.i16(1)
    position = group * tl.u32(32) + entry * tl.u32(4) + payload
    return tl.where(sign < tl.i8(0), -weight, weight), scale, position


@triton.jit
def compute_iq3_xxs(W, X: tl.View[ggml_formats.Q8_K, (K,)], grid):
    acc = tl.state(tl.f32, [PARTIALS], init=0)
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        x = tl.load(X[block])
        weight, scale, position = decode_iq3_xxs(w, grid, None, True)
        products = tl.mac_groups(tl.u8(weight) + tl.u8(128), x.q[position], n=4, into=tl.i32)
        scaled = products * tl.i32(scale)
        groups = tl.reshape(scaled, (4, PARTIALS), axes=("segment", "PARTIALS"),
                            order=("group", "entry", "payload"))
        integer = tl.sum(groups, 0)
        correction_group = tl.arange(0, PARTIALS, dtype=tl.u32, axis="PARTIALS")
        correction_scale = tl.i32(w.metadata[correction_group // tl.u32(2)] >> tl.u32(28)) * tl.i32(2) + tl.i32(1)
        integer -= tl.i32(128) * tl.i32(x.bsum[correction_group]) * correction_scale
        acc += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(integer)
    return tl.f32(0.25) * tl.sum(acc, 0)


@triton.jit(model="weft")
def iq2_s(W: tl.View[ggml_formats.IQ2_S, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          grid: tl.View[ggml_formats.I8X8, (1024, 8)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_iq2_s(W, X, grid))


@triton.jit(model="weft")
def iq3_s(W: tl.View[ggml_formats.IQ3_S, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          grid: tl.View[ggml_formats.I8X4, (512, 4)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_iq3_s(W, X, grid))


@triton.jit(model="weft")
def iq3_xxs(W: tl.View[ggml_formats.IQ3_XXS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
            grid: tl.View[ggml_formats.I8X4, (256, 4)],
            Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_iq3_xxs(W, X, grid))


@triton.jit(model="weft")
def iq3_xxs_specialized(W: tl.View[ggml_formats.IQ3_XXS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                       grid: tl.View[ggml_formats.I8X4, (256, 4)],
                       Y: tl.View[tl.f32, (1,)], *, K: tl.constexpr):
    tl.store(Y[0], compute_iq3_xxs(W, X, grid))


@triton.jit(model="weft")
def specialized_iq2_s(W: tl.View[ggml_formats.IQ2_S, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
                      grid: tl.View[ggml_formats.I8X8, (1024, 8)], Y: tl.View[tl.f32, (1,)],
                      *, K: tl.constexpr):
    tl.store(Y[0], compute_iq2_s(W, X, grid))
