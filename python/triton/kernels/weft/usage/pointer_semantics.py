"""Ordinary Triton pointer kernels for the optional pointer-semantics workloads."""
import triton
import triton.language as tl


@triton.jit(model="weft")
def vector_add(X, Z, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets) + tl.load(Z + offsets))


@triton.jit(model="weft")
def vector_add_tail(X, Z, Y, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    left = tl.load(X + offsets, mask=mask, other=0.0)
    right = tl.load(Z + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, left + right, mask=mask)


@triton.jit(model="weft")
def vector_add_strided(X, Z, Y, N, SX, SZ, SY, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    left = tl.load(X + offsets * SX, mask=mask, other=0.0)
    right = tl.load(Z + offsets * SZ, mask=mask, other=0.0)
    tl.store(Y + offsets * SY, left + right, mask=mask)


@triton.jit(model="weft")
def row_sum(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + offsets, mask=offsets < N, other=0.0)
    tl.store(Y + row, tl.sum(values, axis=0))


@triton.jit(model="weft")
def row_max(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + offsets, mask=offsets < N, other=-float("inf"))
    tl.store(Y + row, tl.max(values, axis=0))


@triton.jit(model="weft")
def row_argmax(X, Y, Index, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + offsets, mask=offsets < N, other=-float("inf"))
    maximum, index = tl.max(values, axis=0, return_indices=True, return_indices_tie_break_left=True)
    tl.store(Y + row, maximum)
    tl.store(Index + row, index)


@triton.jit
def _multiply_values(left, right):
    return left * right


@triton.jit(model="weft")
def row_product(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + offsets, mask=offsets < N, other=1.0)
    tl.store(Y + row, tl.reduce(values, 0, _multiply_values))


@triton.jit(model="weft")
def gather_join_split(X, I0, I1, Y0, Y1, YNeg, M: tl.constexpr, N: tl.constexpr,
                      G0: tl.constexpr, G1: tl.constexpr):
    rows = tl.arange(0, M)
    columns = tl.arange(0, N)
    selected_rows = tl.arange(0, G0)
    selected_columns = tl.arange(0, G1)
    values = tl.load(X + rows[:, None] * N + columns[None, :])
    indices0 = tl.load(I0 + selected_rows[:, None] * N + columns[None, :])
    indices1 = tl.load(I1 + rows[:, None] * G1 + selected_columns[None, :])
    by_row = tl.gather(values, indices0, axis=0)
    by_column = tl.gather(values, indices1, axis=1)
    by_negative_axis = tl.gather(values, indices1, axis=-1)
    restored_column, restored_negative = tl.split(tl.join(by_column, by_negative_axis + 0.5))
    tl.store(Y0 + selected_rows[:, None] * N + columns[None, :], by_row)
    tl.store(Y1 + rows[:, None] * G1 + selected_columns[None, :], restored_column)
    tl.store(YNeg + rows[:, None] * G1 + selected_columns[None, :], restored_negative)


@triton.jit(model="weft")
def row_softmax(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + offsets, mask=offsets < N, other=-float("inf"))
    numerator = tl.exp(values - tl.max(values, axis=0))
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(Y + row * N + offsets, result, mask=offsets < N)


@triton.jit(model="weft")
def softmax_rows(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for row in range(M):
        values = tl.load(X + row * N + offsets, mask=offsets < N, other=-float("inf"))
        numerator = tl.exp(values - tl.max(values, axis=0))
        result = numerator / tl.sum(numerator, axis=0)
        tl.store(Y + row * N + offsets, result, mask=offsets < N)


@triton.jit(model="weft")
def dot_accumulate(A, B, Acc, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, M)
    columns = tl.arange(0, N)
    reduction = tl.arange(0, K)
    left = tl.load(A + rows[:, None] * K + reduction[None, :])
    right = tl.load(B + reduction[:, None] * N + columns[None, :])
    initial = tl.load(Acc + rows[:, None] * N + columns[None, :])
    result = tl.dot(left, right, initial)
    tl.store(Y + rows[:, None] * N + columns[None, :], result)


@triton.jit
def _add_bias(value, bias):
    return value + bias


@triton.jit(model="weft")
def loop_carry_helper(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(X + offsets)
    for index in range(3):
        value = _add_bias(value, 1.0)
    index = 0
    while index < 2:
        value = value * 0.5
        index += 1
    tl.store(Y + offsets, value)


@triton.jit(model="weft")
def alias_shift(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + offsets, mask=offsets < N, other=0.0)
    tl.store(Y + offsets, values + 1.0, mask=offsets < N)


@triton.jit(model="weft")
def pointer_carry(X, Y, STEPS, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    pointer = X + offsets
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for index in range(STEPS):
        total = total + tl.load(pointer)
        pointer = pointer + BLOCK
    tl.store(Y + offsets, total)


@triton.jit(model="weft")
def pointer_select(X, Z, Y, SELECT_X, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    pointer = X if SELECT_X else Z
    tl.store(Y + offsets, tl.load(pointer + offsets))


@triton.jit(model="weft")
def boolean_not(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, mask=offsets < N, other=False)
    tl.store(Y + offsets, value == False, mask=offsets < N)


@triton.jit(model="weft")
def carried_flat_sum(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, 2)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    for row in range(M):
        total = tl.full((2, BLOCK), 0, tl.float32)
        for start in range(0, N, 2 * BLOCK):
            values = tl.load(X + row * N + start + offsets, start + offsets < N, other=0)
            total += values
        tl.store(Y + row, tl.sum(total))
