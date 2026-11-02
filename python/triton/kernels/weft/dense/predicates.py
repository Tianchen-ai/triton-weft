"""Shared conditions over ordinary Weft numeric values and explicit outputs."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", strided=("X", "Residual", "Y", "Z"),
             alias_groups={"X": "input", "Residual": "input", "Y": "first", "Z": "second"})
def masked_affine(X: tl.View[tl.f32, (M, N)], Residual: tl.View[tl.f32, (M, N)],
                  Y: tl.View[tl.f32, (M, N)], Z: tl.View[tl.f32, (M, N)],
                  lower: tl.f32 = -1.0, upper: tl.f32 = 1.0, scale: tl.f32 = 0.75, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            residual = tl.load(Residual[row, part])
            selected = (x >= lower) & (x <= upper)
            tl.store(Y[row, part], tl.select(selected, x * scale + residual, residual))
            tl.store(Z[row, part], tl.select(selected, x, 0.0))


@triton.jit(model="weft", strided=("X", "Ids", "Scores", "SelectedIds"),
             alias_groups={"X": "input", "Ids": "input", "Scores": "scores", "SelectedIds": "ids"})
def threshold_indices(X: tl.View[tl.f32, (M, N)], Ids: tl.View[tl.i32, (M, N)],
                      Scores: tl.View[tl.f32, (M, N)], SelectedIds: tl.View[tl.i32, (M, N)],
                      threshold: tl.f32 = 0.25, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            value = tl.load(X[row, part])
            identity = tl.load(Ids[row, part])
            selected = value >= threshold
            tl.store(Scores[row, part], tl.select(selected, value, 0.0))
            tl.store(SelectedIds[row, part], tl.select(selected, identity, tl.i32(-1)))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def logical_gate(X: tl.View[tl.f32, (M, N)], R: tl.View[tl.f32, (M, N)],
                 Y: tl.View[tl.f32, (M, N)], Z: tl.View[tl.f32, (M, N)], enabled: tl.i1, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = ((x > r) ^ (r > 0.0)) & enabled
            inverse = selected ^ tl.i1(True)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(inverse, x, r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def signed_choices(X: tl.View[tl.i32, (M, N)], R: tl.View[tl.i32, (M, N)],
                   Y: tl.View[tl.i32, (M, N)], Z: tl.View[tl.i32, (M, N)], threshold: tl.i32, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = (x >= threshold) | (r < x)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, tl.i32(-7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def unsigned_choices(X: tl.View[tl.u32, (M, N)], R: tl.View[tl.u32, (M, N)],
                     Y: tl.View[tl.u32, (M, N)], Z: tl.View[tl.u32, (M, N)], threshold: tl.u32, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = (x >= threshold) | (r > x)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, tl.u32(7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def double_choices(X: tl.View[tl.f64, (M, N)], R: tl.View[tl.f64, (M, N)],
                   Y: tl.View[tl.f64, (M, N)], Z: tl.View[tl.f64, (M, N)], threshold: tl.f64, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = (x >= threshold) | (r > x)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, tl.f64(-7.0), r))


@triton.jit(model="weft", strided=("X", "Predicate", "Y"),
             alias_groups={"X": "input", "Predicate": "predicate", "Y": "output"})
def observed_predicate(X: tl.View[tl.f32, (M, N)], Predicate: tl.View[tl.i1, (M, N)],
                       Y: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            selected = x > 0.0
            tl.store(Predicate[row, part], selected)
            tl.store(Y[row, part], tl.select(selected, x, 0.0))


@triton.jit(model="weft", strided=("X", "R", "Mask", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Mask": "input", "Y": "first", "Z": "second"})
def loaded_gate(X: tl.View[tl.f32, (M, N)], R: tl.View[tl.f32, (M, N)],
                Mask: tl.View[tl.i1, (M, N)], Y: tl.View[tl.f32, (M, N)], Z: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = tl.load(Mask[row, part]) & (x >= 0.0)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, 1.0, x))


@triton.jit(model="weft", strided=("X", "R", "Mask", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Mask": "input", "Y": "first", "Z": "second"})
def loaded_choices(X: tl.View[tl.f32, (M, N)], R: tl.View[tl.f32, (M, N)],
                   Mask: tl.View[tl.i1, (M, N)], Y: tl.View[tl.f32, (M, N)], Z: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = tl.load(Mask[row, part])
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, -1.0, x))


@triton.jit(model="weft", strided=("X", "Mask", "Selected", "Y", "Z"),
             alias_groups={"X": "input", "Mask": "input", "Selected": "predicate", "Y": "first", "Z": "second"})
def observed_gate(X: tl.View[tl.f32, (M, N)], Mask: tl.View[tl.i1, (M, N)],
                  Selected: tl.View[tl.i1, (M, N)], Y: tl.View[tl.f32, (M, N)], Z: tl.View[tl.f32, (M, N)], *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            selected = tl.load(Mask[row, part]) | (x < 0.0)
            tl.store(Selected[row, part], selected)
            tl.store(Y[row, part], tl.select(selected, x, 0.0))
            tl.store(Z[row, part], tl.select(selected, 1.0, x))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def signed_byte_choices(X: tl.View[tl.i8, (M, N)], R: tl.View[tl.i8, (M, N)],
                        Y: tl.View[tl.i8, (M, N)], Z: tl.View[tl.i8, (M, N)], threshold: tl.i8, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = (x >= threshold) | (r < x)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, tl.i8(-7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def unsigned_half_choices(X: tl.View[tl.u16, (M, N)], R: tl.View[tl.u16, (M, N)],
                          Y: tl.View[tl.u16, (M, N)], Z: tl.View[tl.u16, (M, N)], threshold: tl.u16, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = (x >= threshold) | (r > x)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, tl.u16(7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def signed_wide_choices(X: tl.View[tl.i64, (M, N)], R: tl.View[tl.i64, (M, N)],
                        Y: tl.View[tl.i64, (M, N)], Z: tl.View[tl.i64, (M, N)], threshold: tl.i64, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = (x >= threshold) | (r < x)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, tl.i64(-7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def unsigned_wide_choices(X: tl.View[tl.u64, (M, N)], R: tl.View[tl.u64, (M, N)],
                          Y: tl.View[tl.u64, (M, N)], Z: tl.View[tl.u64, (M, N)], threshold: tl.u64, *, BLOCK: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as part:
            x = tl.load(X[row, part])
            r = tl.load(R[row, part])
            selected = (x >= threshold) | (r > x)
            tl.store(Y[row, part], tl.select(selected, x, r))
            tl.store(Z[row, part], tl.select(selected, tl.u64(7), r))
