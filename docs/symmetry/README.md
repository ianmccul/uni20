# Symmetry And Block-Sparse Documentation

This directory owns quantum-number semantics and the design of the
symmetry-aware block-sparse Tensor path.

## Implemented Foundation

- [Quantum Numbers and Symmetry](qnum.md) describes `Symmetry`, `QNum`,
  `QNumList`, and `BlockSpace`.
- [Bosonic Abelian BlockTensor Prototype](block_tensor_prototype.md) describes
  the implemented order-zero through order-four sparse slice, its independent
  key-coordinate and dense-axis model, zero-copy permutation and bending, and
  the remaining host-only prototype contract.

## Active Design

- [Spaces, Duals, and Tensor Morphisms](spaces_duals_and_morphisms.md) is the
  canonical boundary, space-identity, wire-bending, and contraction model.
- [BlockTensor](block_tensor.md) is the central symmetry-typed container design.
- [Block-Sparse Tensor and Layout](block_sparse_tensor.md) develops the storage
  and layout model refined by `BlockTensor`.
- [Block Coalescing](block_coalescing.md) covers GEMM grouping and placement
  interactions.
- [Axis Labels, Contraction, and Braiding](axis_labels_and_braiding.md) defines
  the intended user-facing axis and permutation policy.
- [Raw Primitives and Symmetric Lowering](raw_primitives_and_lowering.md)
  identifies the primitive operations needed below symmetry-aware algorithms.

The current `BlockTensor` implementation is only the initial sparse host slice;
complete storage, block operations, async ownership, and the DMRG path remain
to be implemented. Loss of symmetry metadata remains a correctness error, not
an acceptable fallback.

## Source Navigation

- [Implemented symmetry foundations](../../src/uni20/symmetry/)
- [Low-level tensor kernels](../../src/uni20/kernel/)
- [Dense tensor operations](../tensor/operations.md)
- [CPU reference linalg kernels](../../src/uni20/linalg/backends/cpu/)
