# `src/uni20/backend/blas`

This directory contains BLAS/LAPACK discovery and backend-library integration.
It owns vendor selection, integer-width configuration, link flags, and the raw
BLAS interface targets used by higher layers.

## Contents

- `backend_blas.hpp`: aggregate include point for BLAS backend support.
- `blas_int.hpp`: shared checked and non-throwing conversion to the configured
  signed BLAS integer ABI.
- `blas_vendor.*`: detected BLAS vendor reporting.
- `mplapack_binary128.hpp`: binary128 hooks for MPLAPACK-backed scalar support.
- `reference/`: raw reference BLAS declarations and interface target.
- `mkl/`: MKL-specific target wiring.
- `openblas/`: OpenBLAS extension target wiring.

## Notes

- This layer should expose external BLAS availability, not tensor or linalg
  policy.
- Keep LP64/ILP64 handling centralized here so callers do not duplicate BLAS
  integer assumptions.
- The generic BLAS wrappers are currently gated to the portable `N/T/C` GEMM
  transpose opcodes. Some providers expose a conjugate-no-transpose extension:
  OpenBLAS defines `CblasConjNoTrans`, and its develop-branch Fortran GEMM
  dispatcher recognizes `R`. This is a provider-specific fast path, not the
  baseline ABI contract. Higher linalg wrappers should decline direct no-copy
  conjugate-only complex GEMM until a prepared wrapper materializes the
  conjugated operand or an explicit backend extension handles it.
