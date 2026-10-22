from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def pack_pairs(tile, workspace, ROWS: tl.constexpr, PK: tl.constexpr,
               SOURCE_ROW: tl.constexpr, PACKED_ROW: tl.constexpr, P: tl.constexpr):
    """Keep each adjacent K pair together and put output rows between pairs."""
    tl.static_assert(PK % 4 == 0)
    if tl.extent(tile, axis=0) == ROWS and tl.extent(tile, axis=1) == PK * 2:
        for start in range(0, PK, 4):
            source = tl.subview(tile, offsets=(0, start * 2), extents=(ROWS, 8))
            pairs = tl.reshape(tl.i16(tl.load(source)), (ROWS, 4, P),
                               axes=(SOURCE_ROW, "pk", "p"), order=(SOURCE_ROW, "k"))
            values = tl.reshape(pairs, (4, ROWS, P),
                                axes=("pk", PACKED_ROW, "p"), order=("pk", SOURCE_ROW, "p"))
            target = tl.subview(workspace, offsets=(start, 0, 0), extents=(4, ROWS, P))
            tl.store(target, values)
    else:
        zeros = tl.state(tl.i16, [PK, ROWS, P], init=0)
        tl.store(workspace, tl.reshape(zeros, (PK, ROWS, P),
                                      axes=("pk", PACKED_ROW, "p"), order=("pk", "rows", "p")))
        values = tl.reshape(tl.i16(tl.load(tile)), (ROWS, PK, P),
                            axes=(SOURCE_ROW, "pk", "p"), order=(SOURCE_ROW, "k"))
        transposed = tl.reshape(values, (PK, ROWS, P),
                                axes=("pk", PACKED_ROW, "p"), order=("pk", SOURCE_ROW, "p"))
        tl.store(workspace, transposed)


@triton.jit(model="weft", alias_groups={"Scratch": "weights", "Activation": "activation", "Y": "output"})
def paired_matmul_s8(X: tl.View[tl.i8, (M, K)], W: tl.View[tl.i8, (N, K)],
                     Y: tl.View[tl.i32, (M, N)], Scratch: tl.View[tl.i16, (PK, BN, P)],
                     Activation: tl.View[tl.i16, (PK, M, P)],
                     *, BM: tl.constexpr, BN: tl.constexpr, PK: tl.constexpr, P: tl.constexpr = 2):
    """Signed-word grouped products; all packing and zeroing is invocation work."""
    tl.static_assert(P == 2)
    first = tl.i1(True)
    with tl.level.blocks(K, extent=PK * 2) as kb:
        with tl.level.rows(M, group=BM) as mb:
            pack_pairs(X[mb, kb], Activation[:, mb, :], BM, PK, "m", "m", P)
        with tl.level.cols(N, group=BN) as nb:
            pack_pairs(W[nb, kb], Scratch, BN, PK, "n", "bn", P)
            with tl.level.rows(M, group=BM) as mb:
                acc = tl.state(tl.i32, [BM, BN], init=0)
                # The odd K tail is already carried by the pair's validity.
                for pair in range((tl.extent(X[mb, kb], axis=1) + 1) // 2):
                    activation = tl.load(Activation[pair, mb, :])
                    weights = tl.reshape(tl.load(Scratch[pair, :, :]), (BN, P),
                                         axes=("n", "p"), order=("bn", "p"))
                    product = tl.mac_pairs(activation, weights, into=tl.i32)
                    acc += tl.reshape(product, (BM, BN), axes=("m", "n"), order=("m", "n", "p"))
                if not first:
                    acc += tl.load(Y[mb, nb])
                tl.store(Y[mb, nb], acc)
        first = tl.i1(False)
    if first:
        with tl.level.rows(M, group=BM) as mb:
            with tl.level.cols(N, group=BN) as nb:
                acc = tl.state(tl.i32, [BM, BN], init=0)
                tl.store(Y[mb, nb], acc)
