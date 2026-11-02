"""Reusable two-chain reduction over independently scaled record contributions."""
from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def grouped_products(left,right,LANES:tl.constexpr):
    if LANES < 8:
        a=tl.reshape(left,(LANES,32//LANES),axes=("lanes","term"),order=("k",))
        b=tl.reshape(right,(LANES,32//LANES),axes=("lanes","term"),order=("k",))
        product=tl.reduce_dot(a,b,over="term",acc_dtype=tl.i32)
        return tl.reshape(product,(LANES,),axes=("k",),order=("lanes",))
    else:
        return tl.mac_groups(left,right,n=32//LANES,into=tl.i32)


@triton.jit
def dual_accumulate(W,X,table,scale,K,CONTRIBUTION:tl.constexpr,LANES:tl.constexpr=8):
    first=tl.state(tl.f32,[LANES],init=0)
    second=tl.state(tl.f32,[LANES],init=0)
    with tl.level.blocks(K,extent=64) as block:
        for part in tl.static_range(2):
            if tl.extent(X[block],axis=0)>=(part+1)*32:
                value=CONTRIBUTION(W[block],X[block],table,scale,part,LANES)
                if part==0:
                    first+=value
                else:
                    second+=value
    return tl.sum(first+second,0)
