from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def pack_quads(tile, workspace, sums, ROWS: tl.constexpr, QK: tl.constexpr,
               SOURCE_ROW: tl.constexpr, PACKED_ROW: tl.constexpr,
               ACTIVATION: tl.constexpr):
    """Author-owned adjacent byte groups; pad the final group explicitly."""
    total = tl.i32(tl.arange(0, ROWS, axis="row")) * tl.i32(0)
    for start in range(0, (tl.extent(tile, axis=1) + 3) // 4, 4):
        source = tl.subview(tile, offsets=(0, start * 4), extents=(ROWS, 16))
        raw = tl.load(source)
        if ACTIVATION:
            packed = raw
            part = tl.sum(tl.i32(raw), 1)
            total += tl.reshape(part, (ROWS,), axes=("row",), order=(SOURCE_ROW,))
        else:
            packed = tl.u8(raw) + tl.u8(128)
        quads = tl.reshape(packed, (ROWS, 4, 4), axes=(SOURCE_ROW, "qk", "byte"),
                           order=(SOURCE_ROW, "k"))
        quads = tl.reshape(quads, (4, ROWS, 4), axes=("qk", PACKED_ROW, "byte"),
                           order=("qk", SOURCE_ROW, "byte"))
        target = tl.subview(workspace, offsets=(start, 0, 0), extents=(4, ROWS, 4))
        if tl.extent(source, axis=1) < 16:
            zero = (tl.arange(0, 4, axis="qk") * 0 +
                    tl.arange(0, ROWS, axis=PACKED_ROW) * 0 + tl.arange(0, 4, axis="byte") * 0)
            if ACTIVATION:
                zeros = tl.i8(zero)
            else:
                zeros = tl.u8(zero) + tl.u8(128)
            tl.store(target, zeros)
        tl.store(target, quads)
    if ACTIVATION:
        tl.store(sums, tl.reshape(total, (ROWS,), axes=(PACKED_ROW,), order=("row",)))


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation",
                                       "Sums": "sums", "Y": "output"})
def quad_matmul_s8(X: tl.View[tl.i8, (M, K)], W: tl.View[tl.i8, (N, K)],
                   Y: tl.View[tl.i32, (M, N)], Scratch: tl.View[tl.u8, (QK, BN, BYTE)],
                   Activation: tl.View[tl.i8, (QK, M, BYTE)], Sums: tl.View[tl.i32, (M,)],
                   *, BM: tl.constexpr = 4, BN: tl.constexpr = 32,
                   QK: tl.constexpr = 64, BYTE: tl.constexpr = 4):
    """Four byte products per group, with an exact once-per-panel bias correction."""
    tl.static_assert(BYTE == 4 and QK % 4 == 0)
    first = tl.i1(True)
    with tl.level.blocks(K, extent=QK * 4) as kb:
        with tl.level.rows(M, group=BM) as mb:
            pack_quads(X[mb, kb], Activation[:, mb, :], Sums[mb], BM, QK, "m", "m", True)
        with tl.level.cols(N, group=BN) as nb:
            pack_quads(W[nb, kb], Scratch, None, BN, QK, "n", "bn", False)
            with tl.level.rows(M, group=BM) as mb:
                correction = tl.load(Sums[mb]) * tl.i32(128)
                acc = tl.state(tl.i32, [BM, BN], init=0)
                for word in range((tl.extent(X[mb, kb], axis=1) + 3) // 4):
                    activation = tl.reshape(tl.load(Activation[word, mb]), (BM, BYTE),
                                            axes=("m", "byte"), order=("M", "BYTE"))
                    weights = tl.reshape(tl.load(Scratch[word]), (BN, BYTE),
                                         axes=("n", "byte"), order=("BN", "BYTE"))
                    product = tl.mac_groups(activation, weights, n=4, into=tl.i32)
                    acc += tl.reshape(product, (BM, BN), axes=("m", "n"), order=("m", "n", "byte"))
                result = acc - correction
                if not first:
                    result += tl.load(Y[mb, nb])
                tl.store(Y[mb, nb], result)
        first = tl.i1(False)
    if first:
        with tl.level.rows(M, group=BM) as mb:
            with tl.level.cols(N, group=BN) as nb:
                empty = tl.state(tl.i32, [BM, BN], init=0)
                tl.store(Y[mb, nb], empty)
