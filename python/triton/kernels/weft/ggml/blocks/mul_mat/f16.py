from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def production_mul_mat_f16(
    W: tl.View[tl.f16, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xh: tl.View[tl.f16, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
 *, KB: tl.constexpr, KC: tl.constexpr, MC: tl.constexpr, MR: tl.constexpr, NC: tl.constexpr, NR: tl.constexpr,
 CONVERT_BLOCK: tl.constexpr = 128):
    for row in range(M):
        with tl.level.blocks(K, extent=CONVERT_BLOCK) as kb:
            value = tl.load(X[row, kb])
            tl.store(Xh[row, kb], tl.f16(value))
    for row in range(M):
        for column in range(N):
            tl.store(Y[row, column], tl.f32(0.0))
    with tl.level.tiles(N, extent=tl.auto("NC")) as nc:
        with tl.level.tiles(K, extent=tl.auto("KC")) as kc:
            wp = tl.stage(tl.load(W[nc, kc]))
            with tl.level.tiles(M, extent=tl.auto("MC")) as mc:
                xp = tl.stage(tl.load(Xh[mc, kc]))
                with tl.level.rows(mc, group=tl.auto("MR")) as mb:
                    with tl.level.cols(nc, group=tl.auto("NR")) as nb:
                        acc = tl.state(tl.f32, [MR, NR], init=tl.load(Y[mb, nb]))
                        with tl.level.blocks(kc, extent=tl.auto("KB")) as kb:
                            weights = wp[nb, kb]
                            activations = xp[mb, kb]
                            acc += tl.dot(activations, weights, over="k", acc_dtype=tl.f32)
                        tl.store(Y[mb, nb], acc)


@triton.jit(model="weft", alias_groups={"Xh": "workspace", "Y": "output"})
def production_mul_mat_f16_disjoint(
    W: tl.View[tl.f16, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xh: tl.View[tl.f16, (M, K)],
    Y: tl.View[tl.f32, (M, N)],
    *, KB: tl.constexpr, KC: tl.constexpr, MC: tl.constexpr, MR: tl.constexpr,
    NC: tl.constexpr, NR: tl.constexpr, CONVERT_BLOCK: tl.constexpr = 128):
    # GGML's prepared activation workspace and output own distinct storage.
    # Keep this deployment contract explicit; the general helper permits aliasing.
    production_mul_mat_f16(W, X, Xh, Y, KB=KB, KC=KC, MC=MC, MR=MR,
                          NC=NC, NR=NR, CONVERT_BLOCK=CONVERT_BLOCK)
