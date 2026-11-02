from __future__ import annotations

from triton.kernels.weft.increments import formats as ggml_formats
from triton.kernels.weft.functions import quantization as qf
from triton.kernels.weft.increments.partial_sums import dual_accumulate,grouped_products
import triton
import triton.language as tl


@triton.jit
def contribution(W,X,table,scale,PART:tl.constexpr,LANES:tl.constexpr):
    w=tl.load(W)
    x=tl.load(tl.subview(X,offsets=(PART*32,),extents=(32,)))
    lane=tl.arange(0,32,dtype=tl.u32,axis="k")
    group=tl.arange(0,LANES,dtype=tl.u32,axis="k")//tl.u32(LANES//2)
    weight=tl.lookup(table,tl.u8(w.q),bounds="in_bounds")
    products=grouped_products(weight[lane+tl.u32(PART*32)],x.q,LANES)
    bias=tl.u8(x.q)*tl.u8(0)+tl.u8(128)
    products-=grouped_products(bias,x.q,LANES)
    low=qf.exponent_scale(scale,w.d[PART*2])
    high=qf.exponent_scale(scale,w.d[PART*2+1])
    exponent=tl.where(group==tl.u32(0),low,high)
    scaled=tl.f32(products)*(tl.f32(x.d)*exponent)
    return tl.reshape(scaled,(LANES,),axes=("LANES",),order=("k",))


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.NVFP4, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    ue4m3_scale: tl.View[tl.f32, (256,)],
    PARTIALS: tl.constexpr = 8,
):
    table=tl.u8(tl.load(codebook))+tl.u8(128)
    return dual_accumulate(W,X,table,ue4m3_scale,K,contribution,PARTIALS)


@triton.jit(model="weft")
def quantized_vec_dot_nvfp4_q8_0(
    W: tl.View[ggml_formats.NVFP4, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    scale: tl.View[tl.f32, (256,)],
    Y: tl.View[tl.f32, (1,)],
    PARTIALS: tl.constexpr = 8,
):
    tl.store(Y[0], compute(W, X, codebook, scale,PARTIALS))
