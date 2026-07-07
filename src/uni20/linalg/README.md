# `src/uni20/linalg`

This directory contains dense linear-algebra front ends and operation
descriptors. It is the policy layer that selects or describes dense operations
before they lower to backend wrappers and kernels.

## Contents

- `linalg.hpp`: public include point for the dense linalg layer.
- `backend_manifest.hpp`: backend availability and dispatch metadata.
- `blas/`: mdspan-to-BLAS-compatible descriptor and direct wrapper helpers.
- `ops/`: operation descriptors such as matrix-operation tags.
- `backends/cpu/`: CPU dense matrix helpers and the current dense matrix
  exponential implementation.
- `backends/lapack/`: LAPACK-backed linalg entry points.
- `backends/cusolver/`: cuSOLVER-backed linalg entry points.

## Notes

- Keep dense linalg APIs separate from matrix-free Krylov APIs in `krylov/`.
- Backend-specific code should stay in `backends/` and call through the lower
  `backend/` and `kernel/` layers where appropriate.
- Scalar-generic code should use Uni20 scalar traits and numeric limits from
  `core/`.
