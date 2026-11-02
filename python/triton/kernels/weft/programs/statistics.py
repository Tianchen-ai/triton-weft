"""Streaming statistics with explicit state and composable block summaries."""
from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def merge_moments(count, mean, m2, block_count, block_mean, block_m2):
    total = count + block_count
    delta = block_mean - mean
    fraction = tl.f32(block_count) / tl.f32(total)
    next_mean = mean + delta * fraction
    next_m2 = m2 + block_m2 + delta * delta * (tl.f32(count) * fraction)
    return total, next_mean, next_m2


@triton.jit(model="weft", strided=("X", "Y"))
def welford_sequential(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, 3)],
                       *, ROWS: tl.constexpr):
    with tl.level.rows(M, group=ROWS) as rows:
        mean = tl.state(tl.f32, [ROWS], init=0)
        m2 = tl.state(tl.f32, [ROWS], init=0)
        count = tl.state(tl.i32, [ROWS], init=0)
        for column in range(N):
            value = tl.load(X[rows, column])
            count += tl.i32(1)
            delta = value - mean
            mean += delta / tl.f32(count)
            m2 += delta * (value - mean)
        tl.store(Y[rows, 0], tl.f32(count))
        tl.store(Y[rows, 1], mean)
        tl.store(Y[rows, 2], m2)


@triton.jit(model="weft", strided=("X", "Y"))
def welford_grid(X:tl.View[tl.f32,(M,N)],Y:tl.View[tl.f32,(M,3)],
                 *,ROWS:tl.constexpr,BLOCK:tl.constexpr,GRAIN:tl.constexpr):
    first=tl.program_id(0)*GRAIN
    count=tl.minimum(tl.i32(GRAIN),tl.i32(M)-first)
    source=tl.subview(X,offsets=(first,0),extents=(count,N))
    output=tl.subview(Y,offsets=(first,0),extents=(count,3))
    welford_blocked(source,output,ROWS=ROWS,BLOCK=BLOCK)


@triton.jit(model="weft", strided=("X", "Y"))
def welford_blocked(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, 3)],
                    *, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    with tl.level.rows(M, group=ROWS) as rows:
        mean = tl.state(tl.f32, [ROWS], init=0)
        m2 = tl.state(tl.f32, [ROWS], init=0)
        count = tl.state(tl.i32, [ROWS], init=0)
        with tl.level.blocks(N, extent=BLOCK) as columns:
            values = tl.load(X[rows, columns])
            block_count = tl.i32(tl.extent(X[rows, columns], 1))
            block_mean = tl.reduce(values, axis="n") / tl.f32(block_count)
            centered = values - block_mean
            block_m2 = tl.reduce(centered * centered, axis="n")
            count, mean, m2 = merge_moments(count, mean, m2, block_count, block_mean, block_m2)
        tl.store(Y[rows, 0], tl.f32(count))
        tl.store(Y[rows, 1], mean)
        tl.store(Y[rows, 2], m2)


# Config supplies source parameters; the product tuner measures candidates.
welford_tuned = triton.autotune(
    configs=[triton.Config({"ROWS": rows, "BLOCK": block}, num_cpu_threads=1)
             for rows, block in [(1,64),(1,256),(4,256)]], key=[]
)(welford_blocked)
