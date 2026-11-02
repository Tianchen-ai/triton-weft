from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit
def compute(W: tl.View[ggml_formats.Q4_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)]):
    result = tl.f32(0.0)
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        x = tl.load(X[kb])
        group = tl.arange(0, 8, dtype=tl.u8, axis="group")
        lane = tl.arange(0, 32, dtype=tl.u8, axis="k")
        coordinate = group * tl.u8(32) + lane
        partial = tl.reduce_dot(
            w.q[coordinate], x.q[coordinate], over="k", acc_dtype=tl.i32
        )
        total = tl.reduce(partial * tl.i32(w.sc[group]), axis="group")
        min_group = tl.arange(0, 8, dtype=tl.u32, axis="min_group")
        minimum = tl.reduce(
            tl.widen(w.m[min_group], tl.i32)
            * (
                tl.widen(x.bsum[min_group * tl.u32(2)], tl.i32)
                + tl.widen(x.bsum[min_group * tl.u32(2) + tl.u32(1)], tl.i32)
            ),
            axis="min_group",
        )
        # Preserve GGML's block scale/update order before the next quantized layer.
        result -= (tl.f32(x.ds) * tl.f32(w.dmin)) * tl.f32(minimum)
        result += (tl.f32(x.ds) * tl.f32(w.d)) * tl.f32(total)
    return result


@triton.jit(model="weft")
def quantized_vec_dot_q4_k_q8_k(
    W: tl.View[ggml_formats.Q4_K, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]
):
    tl.store(Y[0], compute(W, X))


@triton.jit(model="weft")
def quantized_rows_q4_k_q8_k(
    W: tl.View[ggml_formats.Q4_K, (rows, K)], X: tl.View[ggml_formats.Q8_K, (K,)],
    Y: tl.View[tl.f32, (rows,)], BLOCK_ROWS: tl.constexpr = 4
):
    with tl.level.rows(rows, group=BLOCK_ROWS) as rb:
        result = tl.state(tl.f32, [BLOCK_ROWS], init=0)
        with tl.level.blocks(K, extent=256) as kb:
            w, x = tl.load(W[rb, kb]), tl.load(X[kb])
            group = tl.arange(0, 8, dtype=tl.u8, axis="group")
            lane = tl.arange(0, 32, dtype=tl.u8, axis="k")
            coordinate = group * tl.u8(32) + lane
            partial = tl.reduce_dot(w.q[:, coordinate], x.q[coordinate], over="k", acc_dtype=tl.i32)
            total = tl.reduce(partial * tl.i32(w.sc[:, group]), axis="group")
            minimum_group = tl.arange(0, 8, dtype=tl.u32, axis="minimum_group")
            minimum = tl.reduce(tl.i32(w.m[:, minimum_group]) * (
                tl.i32(x.bsum[minimum_group * tl.u32(2)]) +
                tl.i32(x.bsum[minimum_group * tl.u32(2) + tl.u32(1)])), axis="minimum_group")
            result -= (tl.f32(x.ds) * tl.f32(w.dmin)) * tl.f32(minimum)
            result += (tl.f32(x.ds) * tl.f32(w.d)) * tl.f32(total)
        tl.store(Y[rb], result)


@triton.jit(model="weft")
def quantized_rows_window_q4_k_q8_k(
    W: tl.View[ggml_formats.Q4_K, (rows, K)], X: tl.View[ggml_formats.Q8_K, (K,)],
    Y: tl.View[tl.f32, (rows,)], BLOCK_ROWS: tl.constexpr = 2,
    WINDOW_GROUPS: tl.constexpr = 2,
):
    """Bound the shared activation and wide products to a few scale groups."""
    tl.static_assert(WINDOW_GROUPS == 1 or WINDOW_GROUPS == 2 or WINDOW_GROUPS == 4 or WINDOW_GROUPS == 8)
    group = tl.arange(0, WINDOW_GROUPS, dtype=tl.u8, axis="group")
    lane = tl.arange(0, 32, dtype=tl.u8, axis="k")
    with tl.level.rows(rows, group=BLOCK_ROWS) as rb:
        result = tl.state(tl.f32, [BLOCK_ROWS], init=0)
        with tl.level.blocks(K, extent=256) as kb:
            w, x = tl.load(W[rb, kb]), tl.load(X[kb])
            for window in tl.static_range(8 // WINDOW_GROUPS):
                g = group + tl.u8(window * WINDOW_GROUPS)
                coordinate = g * tl.u8(32) + lane
                partial = tl.reduce_dot(w.q[:, coordinate], x.q[coordinate], over="k", acc_dtype=tl.i32)
                contribution = tl.reduce(partial * tl.i32(w.sc[:, g]), axis="group")
                if window == 0:
                    total = contribution
                else:
                    total += contribution
            minimum_group = tl.arange(0, 8, dtype=tl.u32, axis="minimum_group")
            minimum = tl.reduce(tl.i32(w.m[:, minimum_group]) * (
                tl.i32(x.bsum[minimum_group * tl.u32(2)]) +
                tl.i32(x.bsum[minimum_group * tl.u32(2) + tl.u32(1)])), axis="minimum_group")
            result -= (tl.f32(x.ds) * tl.f32(w.dmin)) * tl.f32(minimum)
            result += (tl.f32(x.ds) * tl.f32(w.d)) * tl.f32(total)
        tl.store(Y[rb], result)


@triton.jit(model="weft")
def quantized_rows_group_stream_q4_k_q8_k(
    W: tl.View[ggml_formats.Q4_K, (rows, K)], X: tl.View[ggml_formats.Q8_K, (K,)],
    Y: tl.View[tl.f32, (rows,)], BLOCK_ROWS: tl.constexpr = 2
):
    """Reuse one activation group while accumulating independent output rows."""
    lane = tl.arange(0, 32, dtype=tl.u8, axis="k")
    with tl.level.rows(rows, group=BLOCK_ROWS) as rb:
        result = tl.state(tl.f32, [BLOCK_ROWS], init=0)
        with tl.level.blocks(K, extent=256) as kb:
            x = tl.load(X[kb])
            w = tl.load(W[rb, kb])
            total = tl.state(tl.i32, [BLOCK_ROWS], init=0)
            minimum = tl.state(tl.i32, [BLOCK_ROWS], init=0)
            for group in tl.static_range(8):
                coordinate = tl.u8(group * 32) + lane
                activation = x.q[coordinate]
                bsum = tl.i32(x.bsum[group * 2]) + tl.i32(x.bsum[group * 2 + 1])
                for row in tl.static_range(BLOCK_ROWS):
                    if row < tl.extent(W[rb, kb], 0):
                        partial = tl.reduce_dot(w.q[row, coordinate], activation, over="k", acc_dtype=tl.i32)
                        total[row] = total[row] + partial * tl.i32(w.sc[row, group])
                        minimum[row] = minimum[row] + bsum * tl.i32(w.m[row, group])
            for row in tl.static_range(BLOCK_ROWS):
                if row < tl.extent(W[rb, kb], 0):
                    value = result[row]
                    value -= (tl.f32(x.ds) * tl.f32(w.dmin[row])) * tl.f32(minimum[row])
                    value += (tl.f32(x.ds) * tl.f32(w.d[row])) * tl.f32(total[row])
                    result[row] = value
        tl.store(Y[rb], result)
