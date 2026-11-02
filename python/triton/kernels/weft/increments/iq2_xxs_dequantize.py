from __future__ import annotations
import triton
import triton.language as tl
from triton.kernels.weft.increments.formats import IQ2_XXS,I8X8
from triton.kernels.weft.increments.iq2_core import decode_iq2

@triton.jit(model="weft")
def row_dequantize_iq2_xxs(W:tl.View[IQ2_XXS,(K,)],grid:tl.View[I8X8,(256,8)],
                          signs:tl.View[I8X8,(128,8)],Y:tl.View[tl.f32,(K,)]):
    with tl.level.blocks(K,extent=256) as kb:
        w=tl.load(W[kb])
        values,scale,coordinates=decode_iq2(w,grid,signs)
        decoded=tl.f32(w.d)*tl.f32(0.125)*tl.f32(scale)*tl.f32(values)
        tl.store(Y[kb],tl.reshape(decoded,(256,),axes=("k",),order=("group","entry","payload")))
