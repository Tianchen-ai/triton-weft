from __future__ import annotations

import triton
import triton.language as tl

inf = tl.constexpr(float("inf"))


@triton.jit(model="weft")
def _rowmax(value):
    return tl.reduce(value, op="max", axis="tk")


@triton.jit(model="weft")
def _rowsum(value):
    return tl.reduce(value, op="add", axis="tk")


@triton.jit(model="weft")
def online_flash_attention(
    Q: tl.View[tl.f16, (Tq, D)],
    K: tl.View[tl.f16, (Tk, D)],
    V: tl.View[tl.f16, (Tk, D)],
    O: tl.View[tl.f16, (Tq, D)],
 *, BK: tl.constexpr, BQ: tl.constexpr):
    with tl.level.rows(Tq, group=tl.auto("BQ")) as qb:
        q = tl.stage(tl.load(Q[qb, :]))
        m = tl.state(tl.f32, [BQ], init=-inf)
        l = tl.state(tl.f32, [BQ], init=0)
        o = tl.state(tl.f32, [BQ, D], init=0)
        with tl.level.blocks(Tk, extent=tl.auto("BK")) as kb:
            k = tl.load(K[kb, :])
            v = tl.load(V[kb, :])
            s = tl.dot(q, k, over="d", acc_dtype=tl.f32)
            m_new = tl.maximum(m, _rowmax(s))
            p = tl.exp(s - m_new)
            alpha = tl.exp(m - m_new)
            l = l * alpha + _rowsum(p)
            o = o * alpha + tl.dot(p, v, over="tk")
            m = m_new
        tl.store(O[qb, :], tl.narrow(o / l, tl.f16, rounding="rne", saturation=False))
