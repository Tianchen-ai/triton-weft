from __future__ import annotations
import triton
import triton.language as tl
from triton.kernels.weft.increments.formats import IQ2_XXS,Q8_K,I8X8
from triton.kernels.weft.increments.iq2_core import decode_iq2

@triton.jit(model="weft")
def quantized_vec_dot_iq2_xxs_q8_k(W:tl.View[IQ2_XXS,(K,)],X:tl.View[Q8_K,(K,)],
                                  grid:tl.View[I8X8,(256,8)],signs:tl.View[I8X8,(128,8)],
                                  Y:tl.View[tl.f32,(1,)],GROUPS:tl.constexpr=8,PARTIALS:tl.constexpr=16,
                                  GROUPED:tl.constexpr=True,SIGN_TABLE:tl.constexpr=False):
    total=tl.state(tl.f32,[PARTIALS],init=0)
    with tl.level.blocks(K,extent=256) as kb:
        w=tl.load(W[kb])
        x=tl.load(X[kb])
        if GROUPED:
            for start in tl.static_range(0,8,GROUPS):
                values,sign,scale,coordinates=decode_iq2(w,grid,signs,True,GROUPS,start,SIGN_TABLE)
                activation=x.q[coordinates]
                activation=tl.where(sign<0,-activation,activation)
                pairs=tl.mac_pairs(values,activation,into=tl.i16)
                lane=tl.arange(0,4,dtype=tl.u16,axis="payload")
                scaled=tl.mac_pairs(pairs,scale+tl.i16(lane*0),into=tl.i32)
                groups=tl.reshape(scaled,(GROUPS*8//PARTIALS,PARTIALS),axes=("segment","PARTIALS"),order=("group","entry","payload"))
                total+=tl.f32(tl.sum(groups,0))*(tl.f32(w.d)*tl.f32(x.ds))
        else:
            values,scale,coordinates=decode_iq2(w,grid,signs,False,8,0,SIGN_TABLE)
            partial=tl.reduce_dot(x.q[coordinates],values,over=("entry","payload"),acc_dtype=tl.i32)
            total+=tl.f32(tl.reduce(partial*tl.i32(scale),axis="group"))*(tl.f32(w.d)*tl.f32(x.ds))
    tl.store(Y[0],tl.sum(total,0)*tl.f32(0.125))
