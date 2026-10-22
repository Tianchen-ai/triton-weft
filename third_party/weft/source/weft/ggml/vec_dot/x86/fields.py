import triton
import triton.language as tl


@triton.jit
def select(field, coordinates, ROWS: tl.constexpr):
    if ROWS:
        return field[:, coordinates]
    return field[coordinates]
