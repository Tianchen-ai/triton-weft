from __future__ import annotations
import triton
import triton.language as tl
from triton.kernels.weft.ggml.vec_dot.x86.signs import even_parity_signs

@triton.jit
def decode_iq2(w,grid,signs,POSITIVE:tl.constexpr=False,GROUPS:tl.constexpr=8,START:tl.constexpr=0,
               SIGN_TABLE:tl.constexpr=False):
    group=tl.arange(0,GROUPS,dtype=tl.u16,axis="group")+tl.u16(START)
    entry=tl.arange(0,4,dtype=tl.u16,axis="entry")
    payload=tl.arange(0,8,dtype=tl.u16,axis="payload")
    word=group*tl.u16(4)
    code=tl.u16(w.q[word+(entry>>tl.u16(1))])
    index=(code>>((entry&tl.u16(1))*tl.u16(8)))&tl.u16(255)
    metadata=tl.widen(w.q[word+tl.u16(2)],tl.u32)|(tl.widen(w.q[word+tl.u16(3)],tl.u32)<<tl.u32(16))
    sign_index=tl.u16(metadata>>(tl.u32(entry)*tl.u32(7)))&tl.u16(127)
    values=tl.lookup(grid.values,index*tl.u16(8)+payload,bounds="in_bounds")
    if SIGN_TABLE:
        sign=tl.lookup(signs.values,sign_index*tl.u16(8)+payload,bounds="in_bounds")
    else:
        sign=even_parity_signs(sign_index,payload)
    scale=tl.i16((metadata>>tl.u32(28))*tl.u32(2)+tl.u32(1))
    coordinates=group*tl.u16(32)+entry*tl.u16(8)+payload
    if POSITIVE:
        return tl.u6(values),sign,scale,coordinates
    return values*sign,scale,coordinates
