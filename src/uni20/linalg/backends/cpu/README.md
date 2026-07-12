# `src/uni20/linalg/backends/cpu`

This directory contains CPU dense linalg implementations that do not require a
vendor accelerator backend.

## Contents

- `gemm.hpp`: reference accessor-respecting GEMM kernel.
- `gemv.hpp`: reference accessor-respecting GEMV kernel.
- `dense_matrix.hpp`: small dense matrix container used by CPU linalg routines.
- `matrix_exponential.hpp`, `matrix_exponential.cpp`: adaptive dense matrix
  exponential implementation.

## Notes

- This is the home for CPU-specific dense algorithms that are above raw kernels
  but below public tensor workflows.
- Keep matrix-free Krylov algorithms in `src/uni20/krylov`; only small dense
  projected problems or direct dense linalg belong here.
