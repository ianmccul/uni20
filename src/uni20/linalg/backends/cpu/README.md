# src/uni20/linalg/backends/cpu

This directory contains CPU dense linalg implementations that do not require a
vendor accelerator backend.

## Contents

- `contract.hpp`: host lease acquisition and dispatch for arbitrary-rank
  pairwise contraction.
- `copy.hpp`: reference accessor-respecting element-copy kernel.
- `conjugate_inplace.hpp`: reference accessor-respecting in-place conjugation
  kernel.
- `transform.hpp`: generic accessor-respecting elementwise overwrite and update
  kernels.
- `strided_transform.hpp`: internal CPU executor for compact strided iteration
  plans shared by copy, conjugation, and generic transforms.
- `gemm.hpp`: reference accessor-respecting GEMM kernel.
- `gemv.hpp`: reference accessor-respecting GEMV kernel.
- `reductions.hpp`: shared accessor-respecting reduction traversal, full and
  partial sums, inner products, and scaled sum-of-squares Euclidean norms.
- `matrix_set.hpp`: accessor-respecting structured matrix initialization.
- `dense_matrix.hpp`: small dense matrix container used by CPU linalg routines.
- `matrix_exponential.hpp`, `matrix_exponential.cpp`: adaptive dense matrix
  exponential implementation.
- [`detail/`](detail/): shared same-precision numerical accumulation helpers.

## Notes

- This is the home for CPU-specific dense algorithms that are above raw kernels
  but below public tensor workflows.
- Reference elementwise kernels prioritize direct, deterministic semantics.
  Layout-specific tiling, vectorization, and HPTT-like traversal belong in
  separate optimized backends that may decline unsupported instances.
- Keep matrix-free algorithms in the [Krylov source layer](../../../krylov/);
  only small dense projected problems or direct dense linalg belong here.

## Related Documentation

- [Linalg backend source map](../)
- [Linear algebra documentation](../../../../../docs/linalg/)
- [Krylov algorithms](../../../../../docs/krylov/algorithms.md)
