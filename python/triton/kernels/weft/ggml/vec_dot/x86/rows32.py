from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl


@triton.jit
def compute(W, X, Y, rows, K, CENTER: tl.constexpr, HIGH: tl.constexpr,
            AFFINE: tl.constexpr, SIGNED: tl.constexpr, ROWS: tl.constexpr):
    LANES: tl.constexpr = 8
    with tl.level.rows(rows, group=ROWS) as output:
        partial = tl.state(tl.f32, [ROWS, LANES], init=0)
        correction = tl.state(tl.f32, [ROWS], init=0)
        with tl.level.blocks(K, extent=32) as block:
            w, x = tl.load(W[output, block]), tl.load(X[block])
            weight, activation = w.q, x.q
            if HIGH:
                q = tl.u8(weight) | (tl.u8(w.qh) << 4)
            else:
                q = tl.u8(weight)
            if SIGNED:
                q = q ^ tl.u8(128)
            products = tl.mac_groups(q, activation, n=4, into=tl.i32)
            products = tl.reshape(products, (ROWS, LANES), axes=("rows", "LANES"), order=("rows", "K"))
            if CENTER or SIGNED:
                ones = tl.u8(activation) * tl.u8(0) + tl.u8(1)
                sums = tl.mac_groups(ones, activation, n=4, into=tl.i32)
                sums = tl.reshape(sums, (LANES,), axes=("LANES",), order=("K",))
                products -= tl.i32(CENTER + (128 if SIGNED else 0)) * sums
            scale = tl.f32(w.d) * tl.f32(x.d)
            partial += scale * tl.f32(products)
            if AFFINE:
                correction += tl.f32(w.m) * tl.f32(x.s)
        values = tl.sum(partial, 1) + correction
        tl.store(Y[output], values)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_0(W: tl.View[ggml_formats.Q4_0, (rows, K)], X: tl.View[ggml_formats.Q8_0, (K,)],
         Y: tl.View[tl.f32, (rows,)], ROWS: tl.constexpr = 4):
    compute(W, X, Y, rows, K, 8, False, False, False, ROWS)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q4_1(W: tl.View[ggml_formats.Q4_1, (rows, K)], X: tl.View[ggml_formats.Q8_1, (K,)],
         Y: tl.View[tl.f32, (rows,)], ROWS: tl.constexpr = 4):
    compute(W, X, Y, rows, K, 0, False, True, False, ROWS)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q5_0(W: tl.View[ggml_formats.Q5_0, (rows, K)], X: tl.View[ggml_formats.Q8_0, (K,)],
         Y: tl.View[tl.f32, (rows,)], ROWS: tl.constexpr = 4):
    compute(W, X, Y, rows, K, 16, True, False, False, ROWS)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q5_1(W: tl.View[ggml_formats.Q5_1, (rows, K)], X: tl.View[ggml_formats.Q8_1, (K,)],
         Y: tl.View[tl.f32, (rows,)], ROWS: tl.constexpr = 4):
    compute(W, X, Y, rows, K, 0, True, True, False, ROWS)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def q8_0(W: tl.View[ggml_formats.Q8_0, (rows, K)], X: tl.View[ggml_formats.Q8_0, (K,)],
         Y: tl.View[tl.f32, (rows,)], ROWS: tl.constexpr = 4):
    compute(W, X, Y, rows, K, 0, False, False, True, ROWS)
