from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
from ..quantize import q8_k as quant_q8_k
import triton
import triton.language as tl


@triton.jit(model="weft")
def _iq2_xs_entry_products(w, x, grid, signs, scale_group, entry, payload):
    linear_entry = scale_group * tl.u32(2) + entry
    code = w.q[:, linear_entry]
    entry_offset = scale_group * tl.u32(16) + entry * tl.u32(8)
    metadata = w.scales[:, scale_group // tl.u32(2)]
    return qf.iq2_xs_entry_reduce(
        code,
        metadata,
        x.q[:, entry_offset + tl.u32(payload)],
        grid,
        signs,
        scale_group,
        payload,
    )


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq2_xs(
    W: tl.View[ggml_formats.IQ2_XS, (N, K)],
    X: tl.View[tl.f32, (M, K)],
    Xq: tl.View[ggml_formats.Q8_K, (M, K)],
    grid: tl.View[ggml_formats.I8X8, (512, 8)],
    signs: tl.View[ggml_formats.I8X8, (128, 8)],
    Y: tl.View[tl.f32, (M, N)],
 *, NR: tl.constexpr):
    quant_q8_k.quantize_matrix(X, Xq)
    grid_values = grid.values
    sign_values = signs.values
    with tl.level.rows(M, group=1) as mb:
        with tl.level.cols(N, group=tl.auto("NR")) as nb:
            f32_acc = tl.state(tl.f32, [1, NR], init=tl.f32(0.0))
            with tl.level.blocks(K, extent=256) as kb:
                w = tl.load(W[nb, kb])
                x = tl.load(Xq[mb, kb])
                scale_group = tl.arange(0, 16, dtype=tl.u32, axis="scale_group")
                entry = tl.arange(0, 2, dtype=tl.u32, axis="entry")
                payload = tl.arange(0, 8, dtype=tl.u16, axis="payload")
                block_sum = _iq2_xs_entry_products(
                    w, x, grid_values, sign_values, scale_group, entry, payload
                )
                f32_acc += (
                    tl.f32(0.125)
                    * tl.f32(w.d)
                    * tl.f32(x.ds)
                    * tl.widen(block_sum, tl.f32)
                )
            tl.store(Y[mb, nb], f32_acc)
