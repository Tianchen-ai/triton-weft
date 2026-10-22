from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def transpose_panel(tile, workspace, ROWS: tl.constexpr, COLS: tl.constexpr,
                    SOURCE_ROW: tl.constexpr, PACKED_ROW: tl.constexpr, STEP: tl.constexpr,
                    STORAGE: tl.constexpr = tl.f32, ROUND_F16: tl.constexpr = False):
    tl.static_assert(COLS % STEP == 0)
    if tl.extent(tile, axis=0) == ROWS and tl.extent(tile, axis=1) == COLS:
        for start in range(0, COLS, STEP):
            source = tl.subview(tile, offsets=(0, start), extents=(ROWS, STEP))
            values = tl.load(source)
            if ROUND_F16:
                values = tl.f16(values)
            values = tl.reshape(values, (STEP, ROWS),
                                axes=("bk", PACKED_ROW), order=("k", SOURCE_ROW))
            target = tl.subview(workspace, offsets=(start, 0), extents=(STEP, ROWS))
            tl.store(target, STORAGE(values))
    else:
        values = tl.load(tile)
        if ROUND_F16:
            values = tl.f16(values)
        values = tl.reshape(values, (COLS, ROWS),
                            axes=("bk", PACKED_ROW), order=("k", SOURCE_ROW))
        tl.store(workspace, STORAGE(values))


@triton.jit
def panel_product(X, W, Y, Scratch, Activation, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  ACC: tl.constexpr, ROUND_F16: tl.constexpr = False):
    """Author-owned packing, reuse and accumulator boundaries for row-major inputs."""
    first = tl.i1(True)
    with tl.level.blocks(K, extent=BK) as kb:
        with tl.level.rows(M, group=BM) as mb:
            transpose_panel(X[mb, kb], Activation[:, mb], BM, BK, "m", "m", 16, ACC, ROUND_F16)
        with tl.level.cols(N, group=BN) as nb:
            transpose_panel(W[nb, kb], Scratch, BN, BK, "n", "bn", 8, ACC)
            with tl.level.rows(M, group=BM) as mb:
                acc = tl.state(ACC, [BM, BN], init=0)
                input_tile = X[mb, kb]
                for k in range(tl.extent(input_tile, axis=1)):
                    activation = ACC(tl.load(Activation[k, mb]))
                    weights = ACC(tl.reshape(tl.load(Scratch[k, :]), (BN,), axes=("n",), order=("bn",)))
                    acc += activation * weights
                if not first:
                    acc += tl.load(Y[mb, nb])
                tl.store(Y[mb, nb], acc)
        first = tl.i1(False)
    if first:
        with tl.level.cols(N, group=BN) as nb:
            with tl.level.rows(M, group=BM) as mb:
                empty = tl.state(ACC, [BM, BN], init=0)
                tl.store(Y[mb, nb], empty)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def packed_matmul_f32(X: tl.View[tl.f32, (M, K)], W: tl.View[tl.f32, (N, K)],
                      Y: tl.View[tl.f32, (M, N)], Scratch: tl.View[tl.f32, (BK, BN)],
                      Activation: tl.View[tl.f32, (BK, M)],
                      *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """F32 panels; both transposes are part of each invocation."""
    panel_product(X, W, Y, Scratch, Activation, M, N, K, BM, BN, BK, tl.f32)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def packed_matmul_bf16(X: tl.View[tl.bf16, (M, K)], W: tl.View[tl.bf16, (N, K)],
                       Y: tl.View[tl.f32, (M, N)], Scratch: tl.View[tl.f32, (BK, BN)],
                       Activation: tl.View[tl.f32, (BK, M)],
                       *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Widen once while packing; retain exact BF16 inputs and F32 accumulation."""
    panel_product(X, W, Y, Scratch, Activation, M, N, K, BM, BN, BK, tl.f32)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def packed_matmul_s8(X: tl.View[tl.i8, (M, K)], W: tl.View[tl.i8, (N, K)],
                     Y: tl.View[tl.i32, (M, N)], Scratch: tl.View[tl.i32, (BK, BN)],
                     Activation: tl.View[tl.i32, (BK, M)],
                     *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Signed inputs, widened products and wrapping I32 accumulation."""
    panel_product(X, W, Y, Scratch, Activation, M, N, K, BM, BN, BK, tl.i32)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def packed_projection_f16_f32(W: tl.View[tl.f16, (N, K)], X: tl.View[tl.f32, (M, K)],
                              Y: tl.View[tl.f32, (M, N)], Scratch: tl.View[tl.f32, (BK, BN)],
                              Activation: tl.View[tl.f32, (BK, M)],
                              *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """F16 weights with the original F32 activation, including invocation packing."""
    panel_product(X, W, Y, Scratch, Activation, M, N, K, BM, BN, BK, tl.f32)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def packed_projection_f16(W: tl.View[tl.f16, (N, K)], X: tl.View[tl.f32, (M, K)],
                          Y: tl.View[tl.f32, (M, N)], Scratch: tl.View[tl.f32, (BK, BN)],
                          Activation: tl.View[tl.f32, (BK, M)],
                          *, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Round activation to F16 during invocation packing, then accumulate in F32."""
    panel_product(X, W, Y, Scratch, Activation, M, N, K, BM, BN, BK, tl.f32, True)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def specialized_matmul_f32(X: tl.View[tl.f32, (M, K)], W: tl.View[tl.f32, (N, K)],
                           Y: tl.View[tl.f32, (M, N)], Scratch: tl.View[tl.f32, (BK, BN)],
                           Activation: tl.View[tl.f32, (BK, M)],
                           *, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """The same panel program with an explicit shape-specialized JIT contract."""
    panel_product(X, W, Y, Scratch, Activation, M, N, K, BM, BN, BK, tl.f32)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def specialized_matmul_bf16(X: tl.View[tl.bf16, (M, K)], W: tl.View[tl.bf16, (N, K)],
                            Y: tl.View[tl.f32, (M, N)], Scratch: tl.View[tl.f32, (BK, BN)],
                            Activation: tl.View[tl.f32, (BK, M)],
                            *, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    panel_product(X, W, Y, Scratch, Activation, M, N, K, BM, BN, BK, tl.f32)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def specialized_matmul_s8(X: tl.View[tl.i8, (M, K)], W: tl.View[tl.i8, (N, K)],
                          Y: tl.View[tl.i32, (M, N)], Scratch: tl.View[tl.i32, (BK, BN)],
                          Activation: tl.View[tl.i32, (BK, M)],
                          *, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    panel_product(X, W, Y, Scratch, Activation, M, N, K, BM, BN, BK, tl.i32)
