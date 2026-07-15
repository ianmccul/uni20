# src/uni20/linalg/backends/cpu

This directory contains CPU dense linalg implementations that do not require a
vendor accelerator backend.

## Contents

- `copy.hpp`: reference accessor-respecting element-copy kernel.
- `conjugate_inplace.hpp`: reference accessor-respecting in-place conjugation
  kernel.
- `gemm.hpp`: reference accessor-respecting GEMM kernel.
- `gemv.hpp`: reference accessor-respecting GEMV kernel.
- `matrix_set.hpp`: accessor-respecting structured matrix initialization.
- `dense_matrix.hpp`: small dense matrix container used by CPU linalg routines.
- `matrix_exponential.hpp`, `matrix_exponential.cpp`: adaptive dense matrix
  exponential implementation.

## Notes

- This is the home for CPU-specific dense algorithms that are above raw kernels
  but below public tensor workflows.
- Keep matrix-free algorithms in the [Krylov source layer](../../../krylov/);
  only small dense projected problems or direct dense linalg belong here.

## Related Documentation

- [Linalg backend source map](../)
- [Linear algebra documentation](../../../../../docs/linalg/)
- [Krylov algorithms](../../../../../docs/krylov/algorithms.md)
