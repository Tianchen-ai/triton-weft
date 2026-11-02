"""Matched F32 online attention: sequential and block-state organizations."""
from __future__ import annotations
import triton
import triton.language as tl

@triton.jit(model="weft")
def online_blocked(Q:tl.View[tl.f32,(Tq,D)],K:tl.View[tl.f32,(Tk,D)],
                   V:tl.View[tl.f32,(Tk,D)],Y:tl.View[tl.f32,(Tq,D)],scale:tl.f32,
                   *,BQ:tl.constexpr,BK:tl.constexpr,D:tl.constexpr):
    with tl.level.rows(Tq,group=BQ) as rows:
        query=tl.stage(tl.load(Q[rows,:]))
        maximum=tl.state(tl.f32,[BQ],init=float("-inf"))
        denominator=tl.state(tl.f32,[BQ],init=0)
        output=tl.state(tl.f32,[BQ,D],init=0)
        with tl.level.blocks(Tk,extent=BK) as keys:
            scores=tl.dot(query,tl.load(K[keys,:]),over="d",acc_dtype=tl.f32)*scale
            next_max=tl.maximum(maximum,tl.reduce(scores,op="max",axis="tk"))
            keep=tl.exp(maximum-next_max)
            probability=tl.exp(scores-next_max)
            denominator=denominator*keep+tl.reduce(probability,axis="tk")
            output=output*keep+tl.dot(probability,tl.load(V[keys,:]),over="tk",acc_dtype=tl.f32)
            maximum=next_max
        tl.store(Y[rows,:],output/denominator)

@triton.jit(model="weft")
def online_sequential(Q:tl.View[tl.f32,(Tq,D)],K:tl.View[tl.f32,(Tk,D)],
                      V:tl.View[tl.f32,(Tk,D)],Y:tl.View[tl.f32,(Tq,D)],scale:tl.f32,
                      *,D:tl.constexpr):
    with tl.level.rows(Tq,group=1) as rows:
        query=tl.stage(tl.load(Q[rows,:]))
        output=tl.state(tl.f32,[1,D],init=0)
        maximum=tl.state(tl.f32,[1],init=float("-inf"))
        denominator=tl.state(tl.f32,[1],init=0)
        for key in range(Tk):
            score=tl.reduce_dot(query,tl.load(K[key,:]),over="d")*scale
            next_max=tl.maximum(maximum,score)
            keep=tl.exp(maximum-next_max)
            probability=tl.exp(score-next_max)
            output=output*keep+tl.load(V[key,:])*probability
            denominator=denominator*keep+probability
            maximum=next_max
        tl.store(Y[rows,:],output/denominator)
