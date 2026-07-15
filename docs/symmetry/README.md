# Symmetry And Block-Sparse Documentation

This directory owns quantum-number semantics and the design of the future
symmetry-aware block-sparse Tensor path.

## Implemented Foundation

- [Quantum Numbers and Symmetry](qnum.md) describes `Symmetry`, `QNum`,
  `QNumList`, and `BlockSpace`.

## Active Design

- [BlockTensor](block_tensor.md) is the central symmetry-typed container design.
- [Block-Sparse Tensor and Layout](block_sparse_tensor.md) develops the storage
  and layout model refined by `BlockTensor`.
- [Block Coalescing](block_coalescing.md) covers GEMM grouping and placement
  interactions.
- [Axis Labels, Contraction, and Braiding](axis_labels_and_braiding.md) defines
  the intended user-facing axis and permutation policy.
- [Raw Primitives and Symmetric Lowering](raw_primitives_and_lowering.md)
  identifies the primitive operations needed below symmetry-aware algorithms.

These design documents do not imply that a complete symmetry-aware
`BlockTensor` or DMRG path exists on the current branch. Loss of symmetry
metadata remains a correctness error, not an acceptable fallback.
