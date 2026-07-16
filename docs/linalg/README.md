# Linear Algebra Documentation

This directory covers dense mdspan lowering, BLAS/LAPACK provider adapters,
optional multiprecision support, and linear-algebra design background.

## Implemented Layers And Forward Work

- [Mdspan Linear Algebra Dispatch](mdspan_dispatch.md) tracks implemented
  operation-tag vertical slices and the remaining dense-operation sequence.
- [BLAS/LAPACK Mdspan Wrappers](blas_lapack_wrappers.md) describes the
  provider-ready operand layer below dispatch.
- [Dense BLAS/LAPACK Wrapper Coverage](dense_blas_lapack_coverage.md) inventories
  active dispatched dependencies and the quarantined experimental wrapper
  survey separately from Krylov algorithm behavior.
- [MPLAPACK Binary128](mplapack_binary128.md) is the build and validation guide
  for optional binary128 provider support.

The two adapter documents contain both implemented checkpoints and future work.
For the current user-facing Tensor operation matrix, use
[Tensor Operations](../tensor/operations.md); tests remain the implementation
evidence for exact backend coverage.

## Design And Background

- [Trace as a Dense Reduction](trace_reduction.md) is an active design note for
  lowering trace-like contractions.
- [Scalar Tensors, Host Scalars, and Storage Transfer](../tensor/scalar_tensors_and_transfer.md)
  defines the implemented full-reduction result-residency boundary and the
  forward transfer model.
- [Tensor-Network Linear Algebra API Survey](tensor_network_api_survey.md) is
  background research, not a Uni20 API contract.

## Source Navigation

- [Dense linalg front ends and dispatch](../../src/uni20/linalg/)
- [Tensor-facing linalg operations](../../src/uni20/linalg/ops/)
- [BLAS descriptor lowering](../../src/uni20/linalg/blas/)
- [Operation-tag backends](../../src/uni20/linalg/backends/)
- [Raw BLAS providers](../../src/uni20/backend/blas/)
- [Raw LAPACK providers](../../src/uni20/backend/lapack/)
- [Linear algebra examples](../../examples/linalg/)
