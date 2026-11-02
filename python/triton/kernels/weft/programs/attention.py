from __future__ import annotations

import triton
import triton.language as tl

inf = tl.constexpr(float("inf"))


@triton.jit(model="weft", strided=("X", "Y"))
def softmax(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, N: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("N")) as columns:
            maximum = tl.reduce(tl.load(X[row, columns]), op="max", axis="n")
            value = tl.load(X[row, columns])
            exponential = tl.exp(value - maximum)
            denominator = tl.reduce(exponential, op="add", axis="n")
            inverse = tl.f32(1.0) / denominator
            tl.store(Y[row, columns], exponential * inverse)


@triton.jit(model="weft", strided=("X", "Y"))
def softmax_row(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)], *, N: tl.constexpr):
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("N")) as columns:
            value = tl.load(X[row, columns])
            maximum = tl.reduce(value, op="max", axis="n")
            exponential = tl.exp(value - maximum)
            total = tl.reduce(exponential, op="add", axis="n")
            inverse = tl.f32(1.0) / total
            tl.store(Y[row, columns], exponential * inverse)


@triton.jit(model="weft", strided=("X", "Cos", "Sin", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Cos": "cache", "Sin": "cache"})
def rope_split(X: tl.View[tl.f32, (T, H, 2, P)], Cos: tl.View[tl.f32, (T, P)],
               Sin: tl.View[tl.f32, (T, P)], Y: tl.View[tl.f32, (T, H, 2, P)], *, BLOCK: tl.constexpr):
    for token in range(T):
        for head in range(H):
            with tl.level.blocks(P, extent=tl.auto("BLOCK")) as part:
                first = tl.load(X[token, head, 0, part])
                second = tl.load(X[token, head, 1, part])
                cosine = tl.load(Cos[token, part])
                sine = tl.load(Sin[token, part])
                tl.store(Y[token, head, 0, part], first * cosine - second * sine)
                tl.store(Y[token, head, 1, part], second * cosine + first * sine)


@triton.jit(model="weft", strided=("X", "Cos", "Sin", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Cos": "cache", "Sin": "cache"})
def rope_shared_coefficients(X: tl.View[tl.f32, (T, H, 2, P)], Cos: tl.View[tl.f32, (T, P)],
                             Sin: tl.View[tl.f32, (T, P)], Y: tl.View[tl.f32, (T, H, 2, P)], *, BLOCK: tl.constexpr):
    for token in range(T):
        with tl.level.blocks(P, extent=BLOCK) as part:
            cosine = tl.stage(tl.load(Cos[token, part]))
            sine = tl.stage(tl.load(Sin[token, part]))
            for head in range(H):
                first = tl.load(X[token, head, 0, part])
                second = tl.load(X[token, head, 1, part])
                tl.store(Y[token, head, 0, part], first * cosine - second * sine)
                tl.store(Y[token, head, 1, part], second * cosine + first * sine)


@triton.jit(model="weft", strided=("X", "Cos", "Sin", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Cos": "cache", "Sin": "cache"})
def rope_interleaved(X: tl.View[tl.f32, (T, H, P, 2)], Cos: tl.View[tl.f32, (T, P)],
                     Sin: tl.View[tl.f32, (T, P)], Y: tl.View[tl.f32, (T, H, P, 2)], *, BLOCK: tl.constexpr):
    for token in range(T):
        for head in range(H):
            with tl.level.blocks(P, extent=tl.auto("BLOCK")) as part:
                first = tl.load(X[token, head, part, 0])
                second = tl.load(X[token, head, part, 1])
                cosine = tl.load(Cos[token, part])
                sine = tl.load(Sin[token, part])
                tl.store(Y[token, head, part, 0], first * cosine - second * sine)
                tl.store(Y[token, head, part, 1], second * cosine + first * sine)


@triton.jit(model="weft", strided=("K", "V", "KeyCache", "ValueCache"),
             alias_groups={"K": "input", "V": "input", "KeyCache": "keys", "ValueCache": "values"})
def kv_update(K: tl.View[tl.f32, (B, T, G, D)], V: tl.View[tl.f32, (B, T, G, D)],
               KeyCache: tl.View[tl.f32, (B, S, G, D)], ValueCache: tl.View[tl.f32, (B, S, G, D)],
               start: tl.index, *, BLOCK: tl.constexpr):
    for batch in range(B):
        for token in range(T):
            for head in range(G):
                with tl.level.blocks(D, extent=tl.auto("BLOCK")) as part:
                    tl.store(KeyCache[batch, start + token, head, part], tl.load(K[batch, token, head, part]))
                    tl.store(ValueCache[batch, start + token, head, part], tl.load(V[batch, token, head, part]))


@triton.jit(model="weft", strided=("Q", "K", "V", "Y"),
             alias_groups={"Q": "input", "K": "input", "V": "input", "Y": "output"})
def causal_attention(Q: tl.View[tl.f32, (B, T, H, D)], K: tl.View[tl.f32, (B, S, G, D)],
                      V: tl.View[tl.f32, (B, S, G, D)], Y: tl.View[tl.f32, (B, T, H, D)],
                      start: tl.index, scale: tl.f32, *, D: tl.constexpr):
    for batch in range(B):
        for token in range(T):
            for head in range(H):
                group = head / (H / G)
                with tl.level.blocks(D, extent=tl.auto("D")) as dimensions:
                    query = tl.stage(tl.load(Q[batch, token, head, dimensions]))
                    accumulator = tl.state(tl.f32, [D], init=0)
                    maximum = tl.f32(-inf)
                    denominator = tl.f32(0.0)
                    for key in range(start + token + 1):
                        score = tl.reduce_dot(query, tl.load(K[batch, key, group, dimensions]), over="d") * scale
                        next_max = tl.maximum(maximum, score)
                        keep = tl.exp(maximum - next_max)
                        probability = tl.exp(score - next_max)
                        accumulator = accumulator * keep + tl.load(V[batch, key, group, dimensions]) * probability
                        denominator = denominator * keep + probability
                        maximum = next_max
                    tl.store(Y[batch, token, head, dimensions], accumulator / denominator)
