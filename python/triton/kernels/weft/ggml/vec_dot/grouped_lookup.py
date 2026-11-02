from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit
def lookup_pair(table, first, second, lane):
    half = lane // tl.u16(8)
    entry = first * (tl.u16(1) - half) + second * half
    return tl.lookup(table, entry * tl.u16(8) + lane % tl.u16(8), bounds="in_bounds")


@triton.jit
def compute_iq1_s(W, X, grid, K):
    """Combine the two codebook pairs before each exact scale-group reduction."""
    lane = tl.arange(0, 16, dtype=tl.u16, axis="element")
    table = grid.values
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        main, correction = tl.i32(0), tl.i32(0)
        for group in tl.static_range(8):
            g = tl.u16(group)
            metadata = tl.u16(w.qh[g])
            scale = tl.i32((metadata >> tl.u16(12) & tl.u16(7)) * tl.u16(2) + tl.u16(1))
            partial = tl.i16(lane) * tl.i16(0)
            for pair in tl.static_range(2):
                p = tl.u16(pair)
                first = tl.u16(w.q[g * tl.u16(4) + p * tl.u16(2)])
                first |= (metadata >> (p * tl.u16(6)) & tl.u16(7)) << tl.u16(8)
                second = tl.u16(w.q[g * tl.u16(4) + p * tl.u16(2) + tl.u16(1)])
                second |= (metadata >> (p * tl.u16(6) + tl.u16(3)) & tl.u16(7)) << tl.u16(8)
                weight = lookup_pair(table, first, second, lane)
                activation = x.q[g * tl.u16(32) + p * tl.u16(16) + lane]
                partial += tl.i16(activation) * tl.i16(weight)
            # IQ1_S's fixed GGML codebook is ternary: the two signed-byte
            # products sum to at most 256 per lane before widening reduction.
            main += scale * tl.reduce(tl.i32(partial), axis="element")
            delta = tl.i32((tl.i16(metadata) >> tl.i16(15)) | tl.i16(1))
            correction += scale * delta * (tl.i32(x.bsum[g * tl.u16(2)]) +
                                           tl.i32(x.bsum[g * tl.u16(2) + tl.u16(1)]))
        combined = tl.f32(main) + tl.f32(0.125) * tl.f32(correction)
        result += tl.f32(w.d) * tl.f32(x.ds) * combined
    return result


@triton.jit
def compute_iq1_m(W, X, grid, K, DEFERRED: tl.constexpr = False):
    """Decode GGML's ternary eighth-unit numerators and their scale groups."""
    lane = tl.arange(0, 16, dtype=tl.u16, axis="element")
    table = grid.values
    result = tl.f32(0)
    if DEFERRED:
        result = tl.f32(lane) * tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        sc0 = tl.u16(w.scales[0]) | tl.u16(w.scales[1]) << tl.u16(8)
        sc1 = tl.u16(w.scales[2]) | tl.u16(w.scales[3]) << tl.u16(8)
        sc2 = tl.u16(w.scales[4]) | tl.u16(w.scales[5]) << tl.u16(8)
        sc3 = tl.u16(w.scales[6]) | tl.u16(w.scales[7]) << tl.u16(8)
        bits = sc0 >> tl.u16(12) | (sc1 >> tl.u16(8) & tl.u16(240))
        bits |= (sc2 >> tl.u16(4) & tl.u16(3840)) | (sc3 & tl.u16(61440))
        d = tl.cast(bits, tl.f16, bitcast=True)
        total = tl.i32(0)
        if DEFERRED:
            total = tl.i32(lane) * tl.i32(0)
        for group in tl.static_range(16):
            g = tl.u16(group)
            coordinate = g * tl.u16(16) + lane
            high = tl.u16(w.qh[g])
            first = tl.u16(w.q[g * tl.u16(2)]) | (high & tl.u16(7)) << tl.u16(8)
            second = tl.u16(w.q[g * tl.u16(2) + tl.u16(1)])
            second |= (high >> tl.u16(4) & tl.u16(7)) << tl.u16(8)
            weight = lookup_pair(table, first, second, lane)
            if DEFERRED:
                first_delta = tl.i8(1) - tl.i8(high >> tl.u16(3) & tl.u16(1)) * tl.i8(2)
                second_delta = tl.i8(1) - tl.i8(high >> tl.u16(7) & tl.u16(1)) * tl.i8(2)
                delta = tl.where(lane < tl.u16(8), first_delta, second_delta)
            else:
                shift = tl.u16(3) + (lane // tl.u16(8)) * tl.u16(4)
                delta = tl.i8(1) - tl.i8(high >> shift & tl.u16(1)) * tl.i8(2)
            numerator = tl.i8(weight) * tl.i8(8) + delta
            word = tl.where(g < tl.u16(4), sc0, tl.where(g < tl.u16(8), sc1,
                            tl.where(g < tl.u16(12), sc2, sc3)))
            scale = tl.i32((word >> (g % tl.u16(4) * tl.u16(3)) & tl.u16(7)) * tl.u16(2) + tl.u16(1))
            if DEFERRED:
                # |activation * numerator * scale| <= 128 * 9 * 15 = 17280.
                # Widen each exact i16 group product before adding groups.
                product = tl.i16(x.q[coordinate]) * tl.i16(numerator)
                total += tl.i32(product * tl.i16(scale))
            else:
                total += scale * tl.reduce_dot(x.q[coordinate], numerator, over=("element",), acc_dtype=tl.i32)
        result += (tl.f32(d) * tl.f32(x.ds)) * tl.f32(total)
    if DEFERRED:
        return tl.reduce(result, axis="element") * tl.f32(0.125)
    return result * tl.f32(0.125)


@triton.jit
def compute_iq2_xxs(W, X, grid, signs, K, PAIR_ACCUM: tl.constexpr = False):
    lane = tl.arange(0, 16, dtype=tl.u16, axis="element")
    table, sign_table = grid.values, signs.values
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        total = tl.i32(0)
        for group in tl.static_range(8):
            g = tl.u32(group)
            word1 = tl.u32(w.q[g * tl.u32(4) + tl.u32(2)])
            word1 |= tl.u32(w.q[g * tl.u32(4) + tl.u32(3)]) << tl.u32(16)
            partial = tl.i32(0)
            if PAIR_ACCUM:
                partial = tl.i16(lane) * tl.i16(0)
            for pair in tl.static_range(2):
                p = tl.u32(pair)
                word0 = tl.u16(w.q[g * tl.u32(4) + p])
                weight = lookup_pair(table, word0 & tl.u16(255), word0 >> tl.u16(8), lane)
                first = tl.u16(word1 >> (p * tl.u32(14))) & tl.u16(127)
                second = tl.u16(word1 >> (p * tl.u32(14) + tl.u32(7))) & tl.u16(127)
                sign = lookup_pair(sign_table, first, second, lane)
                activation = x.q[tl.u16(g) * tl.u16(32) + tl.u16(p) * tl.u16(16) + lane]
                if PAIR_ACCUM:
                    # GGML IQ2_XXS entries are at most 43. Two signed-byte
                    # products fit i16 exactly: 2 * 128 * 43 = 11008.
                    partial += tl.i16(activation) * tl.i16(weight * sign)
                else:
                    partial += tl.reduce_dot(activation, weight * sign, over=("element",), acc_dtype=tl.i32)
            if PAIR_ACCUM:
                partial = tl.reduce(tl.i32(partial), axis="element")
            total += tl.i32((word1 >> tl.u32(28)) * tl.u32(2) + tl.u32(1)) * partial
        result += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(total)
    return result * tl.f32(0.125)


@triton.jit(model="weft")
def iq1_s(W: tl.View[ggml_formats.IQ1_S, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          grid: tl.View[ggml_formats.I8X8, (2048, 8)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_iq1_s(W, X, grid, K))


@triton.jit(model="weft")
def iq1_m(W: tl.View[ggml_formats.IQ1_M, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          grid: tl.View[ggml_formats.I8X8, (2048, 8)], Y: tl.View[tl.f32, (1,)],
          *, DEFERRED: tl.constexpr = False):
    tl.store(Y[0], compute_iq1_m(W, X, grid, K, DEFERRED))


@triton.jit(model="weft")
def iq2_xxs(W: tl.View[ggml_formats.IQ2_XXS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
            grid: tl.View[ggml_formats.I8X8, (256, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)],
            Y: tl.View[tl.f32, (1,)], *, PAIR_ACCUM: tl.constexpr = False):
    tl.store(Y[0], compute_iq2_xxs(W, X, grid, signs, K, PAIR_ACCUM))


@triton.jit
def compute_iq2_xs(W, X, grid, signs, K):
    lane = tl.arange(0, 16, dtype=tl.u16, axis="element")
    table, sign_table = grid.values, signs.values
    result = tl.f32(0)
    with tl.level.blocks(K, extent=256) as block:
        w, x = tl.load(W[block]), tl.load(X[block])
        total = tl.i32(0)
        for group in tl.static_range(16):
            g = tl.u16(group)
            first, second = tl.u16(w.q[g * tl.u16(2)]), tl.u16(w.q[g * tl.u16(2) + tl.u16(1)])
            weight = lookup_pair(table, first & tl.u16(511), second & tl.u16(511), lane)
            sign = lookup_pair(sign_table, first >> tl.u16(9), second >> tl.u16(9), lane)
            activation = x.q[g * tl.u16(16) + lane]
            partial = tl.reduce_dot(activation, weight * sign, over=("element",), acc_dtype=tl.i32)
            metadata = tl.u16(w.scales[g // tl.u16(2)])
            scale = (metadata >> (g % tl.u16(2) * tl.u16(4))) & tl.u16(15)
            total += tl.i32(scale * tl.u16(2) + tl.u16(1)) * partial
        result += tl.f32(w.d) * tl.f32(x.ds) * tl.f32(total)
    return result * tl.f32(0.125)


@triton.jit(model="weft")
def iq2_xs(W: tl.View[ggml_formats.IQ2_XS, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
           grid: tl.View[ggml_formats.I8X8, (512, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)],
           Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute_iq2_xs(W, X, grid, signs, K))


@triton.jit(model="weft")
def iq1_s_grid(W: tl.View[ggml_formats.IQ1_S, (rows, K)], X: tl.View[ggml_formats.Q8_K, (K,)],
              grid: tl.View[ggml_formats.I8X8, (2048, 8)], Y: tl.View[tl.f32, (rows,)]):
    row = tl.program_id(0)
    if row < rows:
        tl.store(Y[row], compute_iq1_s(W[row], X, grid, K))


@triton.jit(model="weft")
def iq1_m_grid(W: tl.View[ggml_formats.IQ1_M, (rows, K)], X: tl.View[ggml_formats.Q8_K, (K,)],
              grid: tl.View[ggml_formats.I8X8, (2048, 8)], Y: tl.View[tl.f32, (rows,)]):
    row = tl.program_id(0)
    if row < rows:
        tl.store(Y[row], compute_iq1_m(W[row], X, grid, K))


@triton.jit(model="weft")
def iq2_xxs_grid(W: tl.View[ggml_formats.IQ2_XXS, (rows, K)], X: tl.View[ggml_formats.Q8_K, (K,)],
                grid: tl.View[ggml_formats.I8X8, (256, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)],
                Y: tl.View[tl.f32, (rows,)]):
    row = tl.program_id(0)
    if row < rows:
        tl.store(Y[row], compute_iq2_xxs(W[row], X, grid, signs, K))


@triton.jit(model="weft")
def iq2_xs_grid(W: tl.View[ggml_formats.IQ2_XS, (rows, K)], X: tl.View[ggml_formats.Q8_K, (K,)],
               grid: tl.View[ggml_formats.I8X8, (512, 8)], signs: tl.View[ggml_formats.I8X8, (128, 8)],
               Y: tl.View[tl.f32, (rows,)]):
    row = tl.program_id(0)
    if row < rows:
        tl.store(Y[row], compute_iq2_xs(W[row], X, grid, signs, K))
