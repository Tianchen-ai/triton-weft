from __future__ import annotations

import triton
import triton.language as tl

inf = tl.constexpr(float("inf"))


@triton.jit(model="weft", strided=("Logits", "Indices", "Weights"),
             alias_groups={"Logits": "input", "Indices": "indices", "Weights": "weights"})
def router(Logits: tl.View[tl.f32, (T, E)], Indices: tl.View[tl.i32, (T, K)],
            Weights: tl.View[tl.f32, (T, K)]):
    for token in range(T):
        for slot in range(K):
            best = tl.f32(-inf)
            chosen = tl.index(0)
            for expert in range(E):
                taken = tl.i1(False)
                for previous in range(slot):
                    taken = taken | (tl.load(Indices[token, previous]) == tl.i32(expert))
                if taken == False:
                    score = tl.load(Logits[token, expert])
                    if score > best:
                        best = score
                        chosen = expert
            tl.store(Indices[token, slot], tl.i32(chosen))
            tl.store(Weights[token, slot], best)
        maximum = tl.load(Weights[token, 0])
        total = tl.f32(0.0)
        for slot in range(K):
            value = tl.exp(tl.load(Weights[token, slot]) - maximum)
            tl.store(Weights[token, slot], value)
            total += value
        for slot in range(K):
            tl.store(Weights[token, slot], tl.load(Weights[token, slot]) / total)


@triton.jit(model="weft", strided=("X", "Indices", "Offsets", "Cursor", "Positions", "Grouped"),
             alias_groups={"X": "input", "Indices": "indices", "Offsets": "offsets",
                           "Cursor": "cursor", "Positions": "positions", "Grouped": "grouped"})
def group_tokens(X: tl.View[tl.f32, (T, D)], Indices: tl.View[tl.i32, (T, K)],
                  Offsets: tl.View[tl.i32, (EP,)], Cursor: tl.View[tl.i32, (E,)],
                  Positions: tl.View[tl.i32, (T, K)], Grouped: tl.View[tl.f32, (P, D)], *, BLOCK: tl.constexpr):
    for expert in range(E + 1):
        tl.store(Offsets[expert], tl.i32(0))
    for token in range(T):
        for slot in range(K):
            expert = tl.index(tl.load(Indices[token, slot]))
            tl.store(Offsets[expert + 1], tl.load(Offsets[expert + 1]) + tl.i32(1))
    for expert in range(E):
        tl.store(Offsets[expert + 1], tl.load(Offsets[expert + 1]) + tl.load(Offsets[expert]))
        tl.store(Cursor[expert], tl.load(Offsets[expert]))
    for token in range(T):
        for slot in range(K):
            expert = tl.index(tl.load(Indices[token, slot]))
            position = tl.index(tl.load(Cursor[expert]))
            tl.store(Positions[token, slot], tl.i32(position))
            tl.store(Cursor[expert], tl.i32(position + 1))
            with tl.level.blocks(D, extent=tl.auto("BLOCK")) as part:
                tl.store(Grouped[position, part], tl.load(X[token, part]))


@triton.jit(model="weft", strided=("X", "Weight", "Offsets", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Offsets": "input", "Y": "output"})
def expert_linear(X: tl.View[tl.f32, (P, K)], Weight: tl.View[tl.f32, (E, N, K)],
                    Offsets: tl.View[tl.i32, (EP,)], Y: tl.View[tl.f32, (P, N)], *, ROWS: tl.constexpr):
    for expert in range(E):
        begin = tl.index(tl.load(Offsets[expert]))
        end = tl.index(tl.load(Offsets[expert + 1]))
        if K == 0:
            for row in range(begin, end):
                for out in range(N):
                    tl.store(Y[row, out], tl.f32(0.0))
        else:
            if begin < end:
                count = end - begin
                if count == 1:
                    with tl.level.rows(N, group=1) as out:
                        value = tl.reduce_dot(tl.load(Weight[expert, out]), tl.load(X[begin]), over="k")
                        tl.store(Y[begin, out], value)
                else:
                    with tl.level.rows(N, group=tl.auto("ROWS")) as out:
                        activations = tl.subview(X, offsets=(begin, 0), extents=(count, K))
                        outputs = tl.subview(Y, offsets=(begin, 0), extents=(count, N))
                        value = tl.dot(tl.load(activations), tl.load(Weight[expert, out]), over="k")
                        tl.store(outputs[:, out], value)


@triton.jit(model="weft", strided=("Grouped", "Positions", "Weights", "Y"),
             alias_groups={"Grouped": "input", "Positions": "input", "Weights": "input", "Y": "output"})
def combine(Grouped: tl.View[tl.f32, (P, D)], Positions: tl.View[tl.i32, (T, K)],
             Weights: tl.View[tl.f32, (T, K)], Y: tl.View[tl.f32, (T, D)], *, BLOCK: tl.constexpr):
    for token in range(T):
        with tl.level.blocks(D, extent=tl.auto("BLOCK")) as part:
            accumulator = tl.state(tl.f32, [BLOCK], init=0)
            for slot in range(K):
                position = tl.index(tl.load(Positions[token, slot]))
                accumulator += tl.load(Grouped[position, part]) * tl.load(Weights[token, slot])
            tl.store(Y[token, part], accumulator)
