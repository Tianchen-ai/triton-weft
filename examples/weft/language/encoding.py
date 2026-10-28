"""Declare, prepare and consume packed storage without a GGML dependency.

Run from the repository root: python -m examples.weft.language.encoding
The host explicitly packs two logical nibbles per byte. Encoding describes those
bytes; neither the declaration nor the kernel performs hidden preparation.
"""
from __future__ import annotations

import numpy as np
import triton
import triton.language as tl


@triton.encoding
class PackedNibbles:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 32
    q: tl.u4[32] @ tl.grouped(elements=32) @ tl.bit_layers(elements=16, order=tl.lo_first)


@triton.jit(model="weft")
def unpack(X: tl.View[PackedNibbles, (N,)], Y: tl.View[tl.u8, (N,)]):
    with tl.level.blocks(N, extent=32) as block:
        record = tl.load(X[block])
        tl.store(Y[block], tl.widen(record.q, tl.u8))


def main():
    logical = (np.arange(64, dtype=np.uint8) % 16).reshape(2, 32)
    storage = logical[:, :16] | (logical[:, 16:] << 4)
    # Host preparation and kernel consumption share the declared layout identity.
    x = tl.View(storage, shape=(logical.size,), encoding=PackedNibbles)
    y = np.empty(logical.size, dtype=np.uint8)
    compiled = unpack[(1,)](x, y)
    print("Storage: 2 records, 16 bytes per record, 32 logical u4 values per record")
    print("Packed bytes in the first record:", storage[0])
    print("Unpacked u8 values in the first record:", y[:32])
    print("Storage contract:", compiled.metadata.kernel_abi["arguments"][0]["encoding"])


if __name__ == "__main__":
    main()
