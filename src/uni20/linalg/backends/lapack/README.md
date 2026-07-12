# `src/uni20/linalg/backends/lapack`

This directory contains LAPACK operation-tag backend adapters.

## Contents

- `common.hpp`: checked LAPACK workspace-query conversion helpers.
- `tridiagonal_eigen.hpp`: symmetric tridiagonal eigenvalues and eigenvectors
  through `sterf` and `steqr`.
- `nonsymmetric_eigen.hpp`: real and complex nonsymmetric eigensystems through
  `geev`, including real conjugate-pair unpacking.
- `self_adjoint_eigh.hpp`: real symmetric and complex Hermitian eigensystems
  through `syev`/`heev`.
- `schur.hpp`: real and complex Schur decomposition, real Hessenberg Schur
  reduction, and Schur block/entry reordering through `gees`, `hseqr`, and
  `trexc`.

## Notes

- Each operation should provide `kernel_accepts_types(LapackBackend, ...)` and
  `try_kernel(LapackBackend, ...)`, then call the raw wrappers under
  `src/uni20/backend/lapack`.
- Return a non-success `KernelAttempt` only for clean preflight decline. LAPACK
  `INFO` failures after a provider call are terminal and must not trigger
  fallback.
- Current update-matrix paths require a directly addressable column-major
  mdspan. Unsupported layouts decline before any operand is modified.
- Keep copy/materialization behavior explicit when adapting tensor views to
  LAPACK-compatible storage.
