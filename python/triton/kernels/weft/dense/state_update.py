from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", strided=("State", "Delta"))
def state_update(State: tl.View[tl.f32, (M, N)], Delta: tl.View[tl.f32, (M, N)],
                 decay: tl.f32 = 1.0, *, B: tl.constexpr):
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=tl.auto("B")) as block:
            state = tl.load(State[row, block])
            delta = tl.load(Delta[row, block])
            tl.store(State[row, block], state * decay + delta)


@triton.jit(model="weft", strided=("State",))
def state_scale(State: tl.View[tl.f32, (M, N)], scale: tl.f32 = 1.0, *, B: tl.constexpr):
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=tl.auto("B")) as block:
            tl.store(State[row, block], tl.load(State[row, block]) * scale)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "activation", "Y": "activation"})
def recurrent_accumulate(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)],
                         steps: tl.index, decay: tl.f32 = 0.5, *, BLOCK: tl.constexpr):
    # Alternate an input update and decay with explicit ordered state.
    for row in range(M):
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as block:
            state = tl.state(tl.f32, [BLOCK], init=0)
            iteration = tl.index(0)
            while iteration < steps:
                if iteration % 2 == 0:
                    state = state + tl.load(X[row, block])
                else:
                    state = state * decay
                iteration += 1
            tl.store(Y[row, block], state)


@triton.jit(model="weft", strided=("X", "Y", "Z"),
             alias_groups={"X": "input", "Y": "first", "Z": "second"})
def coupled_recurrence(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                       Z: tl.View[tl.f32, (N,)], steps: tl.index):
    first = tl.state(tl.f32, [N], init=0)
    second = tl.state(tl.f32, [N], init=1)
    iteration = tl.index(0)
    while iteration < steps:
        previous = first
        first = second + tl.load(X)
        if iteration % 2 == 0:
            second = previous * 0.5 + second * 0.25
        else:
            second = previous - second * 0.125
        iteration += 1
    tl.store(Y, first)
    tl.store(Z, second)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def branch_recurrence(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                      steps: tl.index):
    state = tl.state(tl.f32, [N], init=0)
    iteration = tl.index(0)
    while iteration < steps:
        if iteration % 2 == 0:
            state = state + tl.load(X)
        else:
            state = state * 0.5
        iteration += 1
    tl.store(Y, state)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def initialized_recurrence(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                           steps: tl.index):
    state = tl.state(tl.f32, [N], init=tl.load(X))
    for iteration in range(steps):
        state = state * 0.5 + 1.0
    tl.store(Y, state)


@triton.jit(model="weft", strided=("Y",))
def reset_recurrence(Y: tl.View[tl.f32, (N,)], steps: tl.index):
    state = tl.state(tl.f32, [N], init=-1)
    for iteration in range(steps):
        state = tl.state(tl.f32, [N], init=3)
    tl.store(Y, state)


@triton.jit(model="weft", strided=("X", "Y", "Z", "W", "Initial"),
             alias_groups={"X": "activation", "Y": "activation", "Z": "second",
                           "W": "third", "Initial": "snapshot"})
def parallel_carries(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                     Z: tl.View[tl.f32, (N,)], W: tl.View[tl.f32, (N,)],
                     Initial: tl.View[tl.f32, (N,)], steps: tl.index):
    original = tl.load(X)
    first = tl.state(tl.f32, [N], init=original)
    second = tl.state(tl.f32, [N], init=original * 2.0)
    third = tl.state(tl.f32, [N], init=original + 3.0)
    for iteration in range(steps):
        if iteration % 3 == 0:
            previous = first
            first = second
            second = third
            third = previous
        elif iteration % 3 == 1:
            first = second
            third = third + original
        else:
            previous = first
            first = third + original
            third = second
            second = previous
    tl.store(Y, first)
    tl.store(Z, second)
    tl.store(W, third)
    tl.store(Initial, original)
