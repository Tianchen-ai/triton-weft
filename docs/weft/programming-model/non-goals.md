# Author, compiler and application

Use these boundaries when choosing where to implement an optimization.

| Responsibility | Owner | Example |
|---|---|---|
| Algorithm and numerical decomposition | Author DSL | Split an integer sum into groups, then apply a scale |
| Logical blocking and reuse | Author DSL | Share a panel across several output groups |
| Persistent format | Encoding and explicit builder | Interleave weight records during model initialization |
| Local representation | Compiler passes | Pack the same logical Value for a matrix instruction |
| Physical schedule and resources | Compiler passes | Select LMUL, packet width, pipeline depth and spill placement |
| Instance partition and concurrency | Host application / grid | Assign disjoint rows to OpenMP or GGML workers |
| Storage ownership | Host application | Allocate tensors, workspace and persistent state |
| Comparison protocol | Bench | Match shape, reference, numerical contract and timer |

## Where to change a slow program

If the source repeatedly prepares a panel inside an inner Level, move or redesign
that preparation explicitly in the DSL. If a staged panel already has the right
scope but generated code repeatedly decodes it, inspect the physical supply and
resource passes. If native compute is fast but application calls are expensive,
inspect launch, copies and application integration.

## Observable contracts

Input/output layouts, persistent storage, numerical conversion points and
initialization counts are visible program behavior. Local register layouts,
temporary packing and instruction issue order belong to physical compilation.
The [detailed boundary specification](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/programming-model/non-goals.md)
records the exact constraints used by verifier and pass implementations.
