from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as tl


@triton.encoding
class BlockSum32:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    value: tl.i16


@triton.jit
def quantize(X, Y, K, WITH_SUM: tl.constexpr, BLOCKS: tl.constexpr,
             SAVE_SUM: tl.constexpr = False, Sums=None):
    """Independent scale and inverse divisions, as in GGML's CPU program."""
    with tl.level.blocks(K, extent=32 * BLOCKS) as block:
        x = tl.reshape(tl.load(X[block]), (BLOCKS, 32), axes=("record", "lane"), order=("K",))
        maximum = tl.reduce(tl.abs(x), op="max", axis="lane")
        scale = maximum / tl.f32(127)
        nonzero = maximum != tl.f32(0)
        inverse = tl.where(nonzero, tl.f32(127) / tl.where(nonzero, maximum, tl.f32(1)), tl.f32(0))
        q = tl.narrow(x * inverse, tl.i8, rounding="dynamic", saturation=False)
        encoded_scale = tl.f16(scale)
        if WITH_SUM or SAVE_SUM:
            total = tl.reduce(tl.widen(q, tl.i32), op="add", axis="lane")
        if WITH_SUM:
            encoded_sum = tl.f16(tl.f32(total) * scale)
        for record in tl.static_range(BLOCKS):
            if record * 32 < tl.extent(Y[block], axis=0):
                destination = tl.subview(Y[block], offsets=(record * 32,), extents=(32,))
                tl.store(destination.d, encoded_scale[record])
                if WITH_SUM:
                    tl.store(destination.s, encoded_sum[record])
                tl.store(destination.q, tl.reshape(q[record], (32,), axes=("K",), order=("lane",)))
                if SAVE_SUM:
                    sums = tl.subview(Sums[block], offsets=(record * 32,), extents=(32,))
                    tl.store(sums.value, tl.i16(total[record]))


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output"})
def q8_0(X: tl.View[tl.f32, (K,)], Y: tl.View[ggml.Q8_0, (K,)], BLOCKS: tl.constexpr = 4):
    quantize(X, Y, K, False, BLOCKS)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output"})
def q8_1(X: tl.View[tl.f32, (K,)], Y: tl.View[ggml.Q8_1, (K,)], BLOCKS: tl.constexpr = 4):
    quantize(X, Y, K, True, BLOCKS)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output"})
def q8_0_specialized(X: tl.View[tl.f32, (K,)], Y: tl.View[ggml.Q8_0, (K,)], *,
                     K: tl.constexpr, BLOCKS: tl.constexpr = 4):
    quantize(X, Y, K, False, BLOCKS)
