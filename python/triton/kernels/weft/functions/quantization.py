from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def symmetric_integer(q, d, zero: int = 0):
    return tl.f32(tl.i32(q) - tl.i32(zero)) * tl.f32(d)


@triton.jit(model="weft")
def min_affine(q, d, minimum):
    return tl.f32(tl.i32(q)) * tl.f32(d) + tl.f32(minimum)


@triton.jit(model="weft")
def k_superblock(q, scale, d, minimum, dmin):
    scaled = tl.f32(d) * tl.f32(tl.i32(scale))
    minimum_scaled = tl.f32(dmin) * tl.f32(tl.i32(minimum))
    return scaled * tl.f32(tl.i32(q)) - minimum_scaled


@triton.jit(model="weft")
def ternary_radix(q, d):
    return tl.f32(tl.i32(q)) * tl.f32(d)


@triton.jit(model="weft")
def iq_codebook(q, scale):
    return tl.f32(q) * tl.f32(scale)


@triton.jit(model="weft")
def fp4_codebook(q, scale):
    return tl.f32(tl.i32(q)) * tl.f32(scale)


@triton.jit(model="weft")
def extract_bits(value, shift, width: int = 1):
    mask = (1 << width) - 1
    return tl.u32(value) >> tl.u32(shift) & tl.u32(mask)


@triton.jit(model="weft")
def nonlinear_lookup(codebook, code):
    return tl.lookup(codebook, tl.u32(code), bounds="in_bounds")


@triton.jit(model="weft")
def small_nonlinear_lookup(codebook, code):
    return tl.lookup(codebook, tl.u8(code), bounds="in_bounds")


@triton.jit(model="weft")
def exponent_scale(table, raw):
    return tl.lookup(table, tl.u32(raw), bounds="in_bounds")


@triton.jit(model="weft")
def radix3_digit_i8(powers, packed, digit):
    power16 = tl.narrow(
        tl.lookup(powers, tl.u32(digit), bounds="in_bounds"),
        tl.u16,
        rounding="rtz",
        saturation=False,
    )
    power = tl.narrow(power16, tl.u8, rounding="rtz", saturation=False)
    wrapped = power * tl.u8(packed)
    decoded = tl.narrow(
        tl.widen(wrapped, tl.u16) * tl.u16(3) >> tl.u16(8),
        tl.u8,
        rounding="rtz",
        saturation=False,
    )
    return tl.i8(decoded) - tl.i8(1)


@triton.jit(model="weft")
def dot_codebook32(q, x, codebook):
    return tl.reduce_dot(small_nonlinear_lookup(codebook, q), x, over="k", acc_dtype=tl.i32)


@triton.jit(model="weft")
def iq2_xs_entry_reduce(code, metadata, activation, grid, signs, scale_group, payload):
    code_bits = tl.u16(code)
    grid_index = code_bits & tl.u16(511)
    sign_index = code_bits >> tl.u16(9)
    weight = tl.lookup(grid, grid_index * tl.u16(8) + payload, bounds="in_bounds")
    sign = tl.lookup(signs, sign_index * tl.u16(8) + payload, bounds="in_bounds")
    partial = tl.reduce_dot(activation, weight * sign, over=("entry", "payload"), acc_dtype=tl.i32)
    scale = tl.i32(
        (tl.widen(metadata, tl.u32) >> scale_group % tl.u32(2) * tl.u32(4) & tl.u32(15))
        * tl.u32(2)
        + tl.u32(1)
    )
    return tl.reduce(partial * scale, axis="scale_group")
