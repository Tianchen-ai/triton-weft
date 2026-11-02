from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def choices(left, right, output, part):
    tl.store(output[0, part], tl.select(left == right, left, right))
    tl.store(output[1, part], tl.select(left != right, left, right))
    tl.store(output[2, part], tl.select(left < right, left, right))
    tl.store(output[3, part], tl.select(left <= right, left, right))
    tl.store(output[4, part], tl.select(left > right, left, right))
    tl.store(output[5, part], tl.select(left >= right, left, right))


@triton.jit(model="weft", strided=("X", "R", "Y"))
def compare_f32(X: tl.View[tl.f32, (N,)], R: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (6, N)], *, BLOCK: tl.constexpr):
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        choices(tl.load(X[part]), tl.load(R[part]), Y, part)


@triton.jit(model="weft", strided=("X", "R", "Y"))
def compare_f64(X: tl.View[tl.f64, (N,)], R: tl.View[tl.f64, (N,)], Y: tl.View[tl.f64, (6, N)], *, BLOCK: tl.constexpr):
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        choices(tl.load(X[part]), tl.load(R[part]), Y, part)


@triton.jit(model="weft", strided=("X", "R", "Y"))
def compare_i32(X: tl.View[tl.i32, (N,)], R: tl.View[tl.i32, (N,)], Y: tl.View[tl.i32, (6, N)], *, BLOCK: tl.constexpr):
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        choices(tl.load(X[part]), tl.load(R[part]), Y, part)


@triton.jit(model="weft", strided=("X", "R", "Y"))
def compare_u32(X: tl.View[tl.u32, (N,)], R: tl.View[tl.u32, (N,)], Y: tl.View[tl.u32, (6, N)], *, BLOCK: tl.constexpr):
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        choices(tl.load(X[part]), tl.load(R[part]), Y, part)


@triton.jit(model="weft", strided=("X", "Y"))
def compare_scalar_f32(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (6, N)], pivot: tl.f32, *, BLOCK: tl.constexpr):
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        choices(pivot, tl.load(X[part]), Y, part)


@triton.jit(model="weft", strided=("X", "Y"))
def compare_scalar_f64(X: tl.View[tl.f64, (N,)], Y: tl.View[tl.f64, (6, N)], pivot: tl.f64, *, BLOCK: tl.constexpr):
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        choices(pivot, tl.load(X[part]), Y, part)


@triton.jit(model="weft", strided=("X", "Y"))
def compare_scalar_i32(X: tl.View[tl.i32, (N,)], Y: tl.View[tl.i32, (6, N)], pivot: tl.i32, *, BLOCK: tl.constexpr):
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        choices(pivot, tl.load(X[part]), Y, part)


@triton.jit(model="weft", strided=("X", "Y"))
def compare_scalar_u32(X: tl.View[tl.u32, (N,)], Y: tl.View[tl.u32, (6, N)], pivot: tl.u32, *, BLOCK: tl.constexpr):
    with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
        choices(pivot, tl.load(X[part]), Y, part)
