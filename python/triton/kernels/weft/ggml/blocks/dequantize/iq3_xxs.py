from __future__ import annotations

from triton.integrations import ggml_formats
from ....functions import quantization as qf
import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def row_dequantize_iq3_xxs(
    W: tl.View[ggml_formats.IQ3_XXS, (K,)],
    grid: tl.View[tl.i8, (1024,)],
    signs: tl.View[tl.i8, (1024,)],
    Y: tl.View[tl.f32, (K,)],
):
    with tl.level.blocks(K, extent=256) as kb:
        w = tl.load(W[kb])
        group = tl.index(0)
        with tl.level.subtiles(kb, extent=32) as group_point:
            metadata = w.metadata[group]
            subscale = tl.f32(0.5) + tl.f32(qf.extract_bits(metadata, 28, 4))
            scale = tl.f32(w.d) * subscale * tl.f32(0.5)
            code = tl.index(0)
            with tl.level.subtiles(group_point, extent=4) as code_point:
                payload = tl.arange(0, 4, dtype=tl.u32, axis="k")
                entry = code // tl.index(2)
                code_in_entry = code % tl.index(2)
                storage_coordinate = group * tl.index(8) + code
                grid_index = tl.u32(w.q[storage_coordinate])
                sign_index = qf.extract_bits(metadata, tl.u32(entry) * tl.u32(7), 7)
                grid_value = qf.nonlinear_lookup(grid, grid_index * tl.u32(4) + payload)
                sign = qf.nonlinear_lookup(
                    signs,
                    sign_index * tl.u32(8)
                    + tl.u32(code_in_entry) * tl.u32(4)
                    + payload,
                )
                tl.store(
                    Y[kb][group_point][code_point],
                    qf.iq_codebook(tl.i32(grid_value) * tl.i32(sign), scale),
                )
                code += tl.index(1)
            group += tl.index(1)
