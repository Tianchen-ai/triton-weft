from __future__ import annotations

import triton
import triton.language as tl

from .activation import silu_value, softplus_value


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "History", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Bias": "input",
                           "History": "history", "Y": "output"})
def causal_conv1d(X: tl.View[tl.f32, (B, T, D)], Weight: tl.View[tl.f32, (D, W)],
                   Bias: tl.View[tl.f32, (D,)], History: tl.View[tl.f32, (B, W, D)],
                   Y: tl.View[tl.f32, (B, T, D)], *, BLOCK: tl.constexpr):
    for batch in range(B):
        for token in range(T):
            with tl.level.blocks(D, extent=tl.auto("BLOCK")) as part:
                for lag in range(W - 1):
                    tl.store(History[batch, lag, part], tl.load(History[batch, lag + 1, part]))
                tl.store(History[batch, W - 1, part], tl.load(X[batch, token, part]))
                accumulator = tl.load(Bias[part])
                for lag in range(W):
                    accumulator += tl.load(History[batch, lag, part]) * tl.load(Weight[part, lag])
                tl.store(Y[batch, token, part], silu_value(accumulator))


@triton.jit(model="weft", strided=("X", "Delta", "Bcoef", "Ccoef", "Alog", "Skip", "Gate", "Bias", "State", "Y"),
             alias_groups={"X": "input", "Delta": "input", "Bcoef": "input", "Ccoef": "input",
                           "Alog": "input", "Skip": "input", "Gate": "input", "Bias": "input",
                           "State": "state", "Y": "output"})
def selective_scan(X: tl.View[tl.f32, (B, T, D)], Delta: tl.View[tl.f32, (B, T, D)],
                     Bcoef: tl.View[tl.f32, (B, T, N)], Ccoef: tl.View[tl.f32, (B, T, N)],
                     Alog: tl.View[tl.f32, (D, N)], Skip: tl.View[tl.f32, (D,)],
                     Gate: tl.View[tl.f32, (B, T, D)], Bias: tl.View[tl.f32, (D,)],
                     State: tl.View[tl.f32, (B, D, N)], Y: tl.View[tl.f32, (B, T, D)], *, N: tl.constexpr):
    for batch in range(B):
        for token in range(T):
            for channel in range(D):
                dt = softplus_value(tl.load(Delta[batch, token, channel]) + tl.load(Bias[channel]))
                value = tl.load(X[batch, token, channel])
                with tl.level.blocks(N, extent=tl.auto("N")) as dimensions:
                    decay = tl.exp(-tl.exp(tl.load(Alog[channel, dimensions])) * dt)
                    state = tl.load(State[batch, channel, dimensions])
                    updated = state * decay + tl.load(Bcoef[batch, token, dimensions]) * (dt * value)
                    tl.store(State[batch, channel, dimensions], updated)
                    total = tl.reduce(updated * tl.load(Ccoef[batch, token, dimensions]), op="add", axis="n")
                    gated = (total + tl.load(Skip[channel]) * value) * silu_value(tl.load(Gate[batch, token, channel]))
                    tl.store(Y[batch, token, channel], gated)
