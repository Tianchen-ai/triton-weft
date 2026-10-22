from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def projection_tiles(X, W, Y, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr, ACC: tl.constexpr):
    with tl.level.tiles(M, extent=BM) as mb:
        with tl.level.tiles(N, extent=BN) as nb:
            acc = tl.state(ACC, [BM, BN], init=0)
            with tl.level.blocks(K, extent=BK) as kb:
                acc += tl.dot(tl.load(X[mb, kb]), tl.load(W[nb, kb]),
                              over="k", acc_dtype=ACC)
            tl.store(Y[mb, nb], acc)


@triton.jit(model="weft")
def matmul_f32(X: tl.View[tl.f32, (M, K)],
               W: tl.View[tl.f32, (N, K)], Y: tl.View[tl.f32, (M, N)],
               *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """F32 X @ W.T using bounded source tiles."""
    projection_tiles(X, W, Y, M, N, K, BM, BN, BK, tl.f32)


@triton.jit(model="weft")
def matmul_bf16_f32(X: tl.View[tl.bf16, (M, K)],
                   W: tl.View[tl.bf16, (N, K)], Y: tl.View[tl.f32, (M, N)],
                   *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """BF16 X @ W.T with F32 products and accumulation."""
    projection_tiles(X, W, Y, M, N, K, BM, BN, BK, tl.f32)


@triton.jit(model="weft")
def matmul_s8_s32(X: tl.View[tl.i8, (M, K)],
                 W: tl.View[tl.i8, (N, K)], Y: tl.View[tl.i32, (M, N)],
                 *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Signed INT8 X @ W.T with wrapping INT32 accumulation."""
    projection_tiles(X, W, Y, M, N, K, BM, BN, BK, tl.i32)
