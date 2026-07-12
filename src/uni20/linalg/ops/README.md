# `src/uni20/linalg/ops`

This directory contains public dense-matrix linalg operation wrappers.
Operation values and diagnostic names are defined centrally in
`../operation_tags.hpp`.

## Contents

- `gemm.hpp`: fixed-output Tensor GEMM front end.
- `gemv.hpp`: fixed-output Tensor GEMV front end.
- `matrix_exponential.hpp`: fixed-output matrix exponential dispatch.
- `matrix_set.hpp`: structured matrix initialization over accessor-respecting
  CPU kernels.
- `nonsymmetric_eigen.hpp`: fixed-output nonsymmetric eigensystem dispatch.
- `schur.hpp`: fixed-output Schur decomposition and block-reordering dispatch.
- `tridiagonal_eigen.hpp`: symmetric tridiagonal eigensystem dispatch.

## Notes

- Operation wrappers should validate linalg-level shape requirements before
  dispatching to a backend.
- Bare mdspans call `dispatch_kernel(selector, operation, operands...)`
  directly; do not add operation-specific aliases for generic dispatch.
- Keep backend-specific implementation details in `../backends`.
