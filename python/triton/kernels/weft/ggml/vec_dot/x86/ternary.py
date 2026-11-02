from __future__ import annotations

from triton.integrations import ggml_formats
import triton
import triton.language as tl
from ...dequantize.radix import ternary_record


@triton.jit
def compute(W, X: tl.View[ggml_formats.Q8_K, (K,)], powers, PACKED_RADIX: tl.constexpr = True,
            RECORDS: tl.constexpr = 2):
    PARTIALS: tl.constexpr = 16
    tl.static_assert(RECORDS == 1 or RECORDS == 2 or RECORDS == 4)
    BLOCKS: tl.constexpr = RECORDS if PACKED_RADIX else 1
    result = tl.state(tl.f32, [PARTIALS], init=0)
    second = tl.state(tl.f32, [PARTIALS], init=0)
    with tl.level.blocks(K, extent=256 * BLOCKS) as block:
        for part in tl.static_range(BLOCKS):
            if part * 256 < tl.extent(W[block], axis=0):
                w = tl.load(tl.subview(W[block], offsets=(part * 256,), extents=(256,)))
                x = tl.load(tl.subview(X[block], offsets=(part * 256,), extents=(256,)))
                if PACKED_RADIX:
                    positive = ternary_record(w)
                else:
                    positive = w.q
                products = tl.mac_groups(positive, x.q, n=4, into=tl.i32)
                grouped = tl.reshape(products, (4, PARTIALS), axes=("segment", "PARTIALS"), order=("k",))
                sums = tl.reshape(tl.i32(x.bsum), (PARTIALS,), axes=("PARTIALS",), order=("k",))
                total = tl.sum(grouped, 0) - sums
                contribution = (tl.f32(w.d) * tl.f32(x.ds)) * tl.f32(total)
                if part == 0:
                    result += contribution
                else:
                    second += contribution
    return tl.sum(result + second, 0)


@triton.jit(model="weft")
def tq1_0(W: tl.View[ggml_formats.TQ1_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)],
          powers: tl.View[tl.u32, (5,)], Y: tl.View[tl.f32, (1,)], *, RECORDS: tl.constexpr = 2):
    tl.store(Y[0], compute(W, X, powers, True, RECORDS))


@triton.jit(model="weft")
def tq2_0(W: tl.View[ggml_formats.TQ2_0, (K,)], X: tl.View[ggml_formats.Q8_K, (K,)], Y: tl.View[tl.f32, (1,)]):
    tl.store(Y[0], compute(W, X, None, False))
