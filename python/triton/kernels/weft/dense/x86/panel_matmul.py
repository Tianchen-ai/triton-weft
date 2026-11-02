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


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def row_blocked_matmul_f32(X: tl.View[tl.f32, (M, K)], W: tl.View[tl.f32, (N, K)],
                           Y: tl.View[tl.f32, (M, N)], Scratch: tl.View[tl.f32, (BK, BN)],
                           Activation: tl.View[tl.f32, (MB, BK, BM)],
                           *, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MB: tl.constexpr,
                           UNROLL: tl.constexpr = 1):
    """Pack contiguous row blocks inside each call; specialize the chosen shape."""
    tl.static_assert(M % BM == 0 and MB == M // BM)
    tl.static_assert(UNROLL > 0)
    first = tl.i1(True)
    with tl.level.blocks(K, extent=BK) as kb:
        row_block = tl.index(0)
        with tl.level.rows(M, group=BM) as mb:
            transpose_small_panels(X[mb, kb], Activation[row_block], BM, BK, "m", "bm")
            row_block += tl.index(1)
        with tl.level.cols(N, group=BN) as nb:
            transpose_small_panels(W[nb, kb], Scratch, BN, BK, "n", "bn")
            row_block = tl.index(0)
            with tl.level.rows(M, group=BM) as mb:
                acc = tl.state(tl.f32, [BM, BN], init=0)
                for start in range(0, tl.extent(X[mb, kb], axis=1), UNROLL):
                    for offset in tl.static_range(UNROLL):
                        k = start + offset
                        if k < tl.extent(X[mb, kb], axis=1):
                            activation = tl.reshape(tl.load(Activation[row_block, k, :]),
                                                     (BM,), axes=("m",), order=("bm",))
                            weights = tl.reshape(tl.load(Scratch[k, :]),
                                                 (BN,), axes=("n",), order=("bn",))
                            acc += activation * weights
                if not first:
                    acc += tl.load(Y[mb, nb])
                tl.store(Y[mb, nb], acc)
                row_block += tl.index(1)
        first = tl.i1(False)
    if first:
        with tl.level.cols(N, group=BN) as nb:
            with tl.level.rows(M, group=BM) as mb:
                tl.store(Y[mb, nb], tl.state(tl.f32, [BM, BN], init=0))


@triton.jit
def transpose_small_panels(tile, workspace, ROWS: tl.constexpr, COLS: tl.constexpr,
                           SOURCE_ROW: tl.constexpr, PACKED_ROW: tl.constexpr):
    R: tl.constexpr = min(ROWS, 8)
    if ROWS % R == 0 and COLS % 8 == 0 and tl.extent(tile, axis=0) == ROWS and tl.extent(tile, axis=1) == COLS:
        for row in tl.static_range(0, ROWS, R):
            for column in range(0, COLS, 8):
                source = tl.subview(tile, offsets=(row, column), extents=(R, 8))
                values = tl.reshape(tl.load(source), (8, R),
                                    axes=("bk", PACKED_ROW), order=("k", SOURCE_ROW))
                destination = tl.subview(workspace, offsets=(column, row), extents=(8, R))
                tl.store(destination, values)
    else:
        transpose_panel(tile, workspace, ROWS, COLS, SOURCE_ROW, PACKED_ROW, 8, tl.f32)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def row_blocked_matmul_grid_f32(X: tl.View[tl.f32, (M, K)], W: tl.View[tl.f32, (N, K)],
                                Y: tl.View[tl.f32, (M, N)],
                                Scratch: tl.View[tl.f32, (PARTS, BK, BN)],
                                Activation: tl.View[tl.f32, (PARTS, MB, BK, BM)],
                                *, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                PARTS: tl.constexpr, GRAIN: tl.constexpr, MB: tl.constexpr,
                                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                                UNROLL: tl.constexpr = 4):
    tl.static_assert(M == PARTS * GRAIN and GRAIN % BM == 0)
    part = tl.program_id(0)
    tl.assume(part >= 0)
    tl.assume(part < PARTS)
    x = tl.subview(X, offsets=(part * GRAIN, 0), extents=(GRAIN, K))
    y = tl.subview(Y, offsets=(part * GRAIN, 0), extents=(GRAIN, N))
    row_blocked_matmul_f32(x, W, y, Scratch[part], Activation[part],
        M=GRAIN, N=N, K=K, BM=BM, BN=BN, BK=BK, MB=MB, UNROLL=UNROLL)
