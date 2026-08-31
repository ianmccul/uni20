# Symmetry And Block-Sparse Documentation

This directory owns quantum-number semantics and the design of the
symmetry-aware block-sparse Tensor path.

## Implemented Foundation

- [Quantum Numbers and Symmetry](qnum.md) describes `Symmetry`, `QNum`,
  `QNumList`, and `BlockSpace`.
- [Bosonic Abelian BlockTensor Prototype](block_tensor_prototype.md) describes
  the implemented key-coordinate and dense-axis model, sparse and complete
  block patterns, host and packed CUDA storage, mapped permutation and
  repartition views, linear operations, generalized adjacent contraction,
  diagonal storage, and staged per-charge SVD and truncation.

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

The implemented bosonic Abelian path now supports host and single-device CUDA
execution, including matrix-free Krylov operations and finite U(1) two-site
DMRG. Broader symmetry categories, multi-device and MPI-distributed block
placement, and wider specialized operation coverage remain active work.

## Source Navigation

- [Implemented symmetry foundations](../../src/uni20/symmetry/)
- [Low-level tensor kernels](../../src/uni20/kernel/)
- [Dense tensor operations](../tensor/operations.md)
- [CPU reference linalg kernels](../../src/uni20/linalg/backends/cpu/)
