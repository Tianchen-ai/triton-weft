from __future__ import annotations

from triton.kernels.weft.increments import formats as ggml_formats
from triton.kernels.weft.functions import quantization as qf
from triton.kernels.weft.increments.partial_sums import dual_accumulate,grouped_products
import triton
import triton.language as tl


@triton.jit
def contribution(W,X,table,scale,PART:tl.constexpr,LANES:tl.constexpr):
    w=tl.load(tl.subview(W,offsets=(PART*32,),extents=(32,)))
    x=tl.load(tl.subview(X,offsets=(PART*32,),extents=(32,)))
    decoded=tl.lookup(table,tl.u8(w.q),bounds="in_bounds")
    products=grouped_products(decoded,x.q,LANES)
    partial=tl.reshape(products,(LANES,),axes=("LANES",),order=("k",))
    return tl.f32(partial)*(tl.f32(x.d)*qf.exponent_scale(scale,w.e))


@triton.jit(model="weft")
def compute(
    W: tl.View[ggml_formats.MXFP4, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    e8m0_scale: tl.View[tl.f32, (256,)],
    PARTIALS: tl.constexpr = 8,
):
    return dual_accumulate(W,X,tl.load(codebook),e8m0_scale,K,contribution,PARTIALS)


@triton.jit(model="weft")
def quantized_vec_dot_mxfp4_q8_0(
    W: tl.View[ggml_formats.MXFP4, (K,)],
    X: tl.View[ggml_formats.Q8_0, (K,)],
    codebook: tl.View[tl.i8, (16,)],
    scale: tl.View[tl.f32, (256,)],
    Y: tl.View[tl.f32, (1,)],
    PARTIALS: tl.constexpr = 8,
):
    tl.store(Y[0], compute(W, X, codebook, scale,PARTIALS))
