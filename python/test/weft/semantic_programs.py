"""Core-model differential program: births, snapshots, aliases and handoff."""
from __future__ import annotations
import triton
import triton.language as tl

@triton.jit
def coupled_update(a,b,delta):
    next_a=tl.i32(tl.i16(a+delta))
    next_b=b+next_a
    return next_a,next_b

@triton.jit(model="weft",strided=("X","Y"))
def semantics_composition(X:tl.View[tl.i32,(M,N)],Init:tl.View[tl.i32,(M,)],Y:tl.View[tl.i32,(M,3)],
                          STEPS:tl.i32,*,ROWS:tl.constexpr,BLOCK:tl.constexpr):
    with tl.level.rows(M,group=ROWS) as rows:
        a=tl.state(tl.i32,[ROWS],init=tl.load(Init[rows]))
        b=tl.state(tl.i32,[ROWS],init=0)
        saved=tl.stage(tl.load(X[rows]))
        visits=tl.i32(0)
        for iteration in range(STEPS):
            with tl.level.blocks(N,extent=BLOCK) as block:
                old=saved[rows,block]
                delta=tl.reduce(old,axis="n")
                a,b=coupled_update(a,b,delta)
                if iteration%2==0:
                    tl.store(X[rows,block],tl.load(X[rows,block])+tl.i32(1))
                else:
                    tl.store(X[rows,block],tl.load(X[rows,block])-tl.i32(1))
                visits+=tl.i32(1)
        tl.store(Y[rows,0],a)
        tl.store(Y[rows,1],b)
        tl.store(Y[rows,2],a*tl.i32(0)+visits)


@triton.jit(model="weft")
def invalid_body_capture(X:tl.View[tl.f32,(M,)],Y:tl.View[tl.f32,(M,)]):
    with tl.level.rows(M,group=4) as rows:
        local=tl.load(X[rows])
        shared=tl.stage(local)
        tl.store(Y[rows],shared)


@triton.jit(model="weft")
def invalid_sibling_capture(X:tl.View[tl.f32,(M,)],Y:tl.View[tl.f32,(M,)]):
    with tl.level.rows(M,group=4) as rows:
        acc=tl.state(tl.f32,[4],init=0)
        shared=tl.stage(acc)
        tl.store(Y[rows],shared)


@triton.jit(model="weft")
def invalid_stage_update(X:tl.View[tl.f32,(M,)],Y:tl.View[tl.f32,(M,)]):
    with tl.level.rows(M,group=4) as rows:
        shared=tl.stage(tl.load(X[rows]))
        shared=shared+tl.f32(1)
        tl.store(Y[rows],shared)


@triton.jit(model="weft")
def static_births(X:tl.View[tl.f32,(M,)],Y:tl.View[tl.f32,(M,)],*,SHARED:tl.constexpr):
    with tl.level.rows(M,group=4) as rows:
        if SHARED:
            shared=tl.stage(tl.load(X[rows]))
        else:
            shared=tl.stage(tl.load(X[rows])*tl.f32(2))
        tl.store(X[rows],tl.load(X[rows])+tl.f32(1))
        tl.store(Y[rows],shared)


@triton.jit
def count_helper_rows(M):
    visits=tl.i32(0)
    with tl.level.rows(M,group=4) as rows:
        visits+=tl.i32(1)
    return visits


@triton.jit(model="weft")
def lexical_helper_extent(X:tl.View[tl.i32,(M,)],Y:tl.View[tl.i32,(2,)],COUNT:tl.i32):
    # The helper's M is its explicit actual argument, not the caller's shape.
    tl.store(Y[0],count_helper_rows(COUNT))
    tl.store(Y[1],count_helper_rows(COUNT//tl.i32(2)))


@triton.jit
def copy_window(X:tl.View[tl.i32,(R,)],Y:tl.View[tl.i32,(R,)]):
    with tl.level.rows(R,group=4) as rows:
        tl.store(Y[rows],tl.load(X[rows])+tl.i32(1))


@triton.jit(model="weft")
def composed_windows(Padding:tl.View[tl.i32,(4,)],X:tl.View[tl.i32,(M,)],
                     Y:tl.View[tl.i32,(M,)],COUNT:tl.i32):
    x=tl.subview(X,offsets=(0,),extents=(COUNT-tl.i32(0),))
    y=tl.subview(Y,offsets=(0,),extents=(COUNT-tl.i32(0),))
    copy_window(x,y)


@triton.jit(model="weft")
def mismatched_windows(X:tl.View[tl.i32,(M,)],Y:tl.View[tl.i32,(M,)],COUNT:tl.i32):
    x=tl.subview(X,offsets=(0,),extents=(COUNT,))
    copy_window(x,Y)


@triton.jit(model="weft")
def static_window_copy(X:tl.View[tl.i32,(M,)],Y:tl.View[tl.i32,(M,)],*,M:tl.constexpr):
    copy_window(X,Y)
