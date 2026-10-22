from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as tl


@triton.jit
def _scaled_product(activation, weight, delta, scale):
    # The GGML IQ1 codebook alphabet is {-1, 0, 1}. Its eighth-unit
    # numerator is in [-9, 9]; with Q8 and odd scales <=15, i16 is exact.
    numerator = tl.i8(weight) * tl.i8(8) + tl.i8(delta)
    product = tl.i16(activation) * tl.i16(numerator)
    return tl.i32(product * tl.i16(scale))


@triton.jit(noinline=True)
def compute_m(W: tl.View[ggml.IQ1_M, (K,)], X: tl.View[ggml.Q8_K, (K,)],
              grid: tl.View[ggml.I8X8, (2048, 8)]):
    """Use the GGML IQ1 codebook and one scaled integer reduction per block."""
    entry = tl.arange(0, 32, dtype=tl.u16, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
    result = tl.f32(0)
    table = grid.values
    with tl.level.blocks(K, extent=256) as block:
        w = tl.load(W[block])
        x = tl.load(X[block])
        sc0 = tl.u32(w.scales[0]) | tl.u32(w.scales[1]) << tl.u32(8)
        sc1 = tl.u32(w.scales[2]) | tl.u32(w.scales[3]) << tl.u32(8)
        sc2 = tl.u32(w.scales[4]) | tl.u32(w.scales[5]) << tl.u32(8)
        sc3 = tl.u32(w.scales[6]) | tl.u32(w.scales[7]) << tl.u32(8)
        bits = sc0 >> tl.u32(12) | (sc1 >> tl.u32(8) & tl.u32(240))
        bits |= (sc2 >> tl.u32(4) & tl.u32(3840)) | (sc3 & tl.u32(61440))
        d = tl.cast(tl.u16(bits), tl.f16, bitcast=True)
        high = tl.u16(w.qh[entry // tl.u16(2)])
        index = tl.u16(w.q[entry])
        entry_shift = entry % tl.u16(2) * tl.u16(4)
        index |= (high >> entry_shift & tl.u16(7)) << tl.u16(8)
        weight = tl.lookup(table, index * tl.u16(8) + payload, bounds="in_bounds")
        coordinate = entry * tl.u16(8) + payload
        word = tl.where(entry < tl.u16(8), tl.u16(sc0),
                        tl.where(entry < tl.u16(16), tl.u16(sc1),
                                 tl.where(entry < tl.u16(24), tl.u16(sc2), tl.u16(sc3))))
        shift = entry // tl.u16(2) % tl.u16(4) * tl.u16(3)
        scale = tl.i8((tl.u8(word >> shift) & tl.u8(7)) * tl.u8(2) + tl.u8(1))
        sign = (high >> (tl.u16(3) + entry_shift)) & tl.u16(1)
        delta = tl.i8(1) - tl.i8(sign) * tl.i8(2)
        scaled = _scaled_product(x.q[coordinate], weight, delta, scale)
        result += (tl.f32(d) * tl.f32(x.ds)) * tl.f32(tl.sum(scaled))
    return result * tl.f32(0.125)


@triton.jit(model="weft")
def dot_m(W: tl.View[ggml.IQ1_M, (K,)], X: tl.View[ggml.Q8_K, (K,)],
          grid: tl.View[ggml.I8X8, (2048, 8)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_m(W, X, grid))


@triton.jit(noinline=True)
def compute_m_windowed(W: tl.View[ggml.IQ1_M, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                       grid: tl.View[ggml.I8X8, (2048, 8)], ENTRIES: tl.constexpr):
    tl.static_assert(ENTRIES == 4 or ENTRIES == 8 or ENTRIES == 16)
    entry_lane = tl.arange(0, ENTRIES, dtype=tl.u16, axis="entry")
    payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
    lane = tl.arange(0, 16, dtype=tl.u16, axis="lane")
    result = tl.f32(0)
    table = grid.values
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        sc0 = tl.u16(w.scales[0]) | tl.u16(w.scales[1]) << tl.u16(8)
        sc1 = tl.u16(w.scales[2]) | tl.u16(w.scales[3]) << tl.u16(8)
        sc2 = tl.u16(w.scales[4]) | tl.u16(w.scales[5]) << tl.u16(8)
        sc3 = tl.u16(w.scales[6]) | tl.u16(w.scales[7]) << tl.u16(8)
        bits = sc0 >> tl.u16(12) | (sc1 >> tl.u16(8) & tl.u16(240))
        bits |= (sc2 >> tl.u16(4) & tl.u16(3840)) | (sc3 & tl.u16(61440))
        d = tl.cast(bits, tl.f16, bitcast=True)
        total = tl.i32(lane) * tl.i32(0)
        for part in tl.static_range(32 // ENTRIES):
            entry = entry_lane + tl.u16(part * ENTRIES)
            high = tl.u16(w.qh[entry // tl.u16(2)])
            shift = entry % tl.u16(2) * tl.u16(4)
            index = tl.u16(w.q[entry]) | ((high >> shift & tl.u16(7)) << tl.u16(8))
            weight = tl.lookup(table, index * tl.u16(8) + payload, bounds="in_bounds")
            delta = tl.i8(1) - tl.i8(high >> (tl.u16(3) + shift) & tl.u16(1)) * tl.i8(2)
            numerator = tl.i8(weight) * tl.i8(8) + delta
            product = tl.i16(x.q[entry * tl.u16(8) + payload]) * tl.i16(numerator)
            linear = tl.reshape(product, (ENTRIES * 8,), axes=("linear",), order=("entry", "payload"))
            for group in tl.static_range(ENTRIES // 2):
                if part * ENTRIES // 2 + group < 4:
                    word = sc0
                elif part * ENTRIES // 2 + group < 8:
                    word = sc1
                elif part * ENTRIES // 2 + group < 12:
                    word = sc2
                else:
                    word = sc3
                scale = tl.i16((word >> tl.u16(((part * ENTRIES // 2 + group) % 4) * 3) & tl.u16(7)) * tl.u16(2) + tl.u16(1))
                total += tl.i32(linear[tl.u16(group * 16) + lane]) * tl.i32(scale)
        result += (tl.f32(d) * tl.f32(x.ds)) * tl.f32(tl.reduce(total, axis="lane"))
    return result * tl.f32(0.125)


@triton.jit(model="weft")
def dot_m_windowed(W: tl.View[ggml.IQ1_M, (K,)], X: tl.View[ggml.Q8_K, (K,)],
                   grid: tl.View[ggml.I8X8, (2048, 8)], Y: tl.View[tl.f32, (1,)], *, ENTRIES: tl.constexpr = 16):
    tl.store(Y[0], compute_m_windowed(W, X, grid, ENTRIES))
