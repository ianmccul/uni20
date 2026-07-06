# `src/uni20/backend`

This directory contains low-level wrappers for external backend libraries. These
headers should expose library capabilities and calling conventions without
owning tensor semantics, symmetry metadata, or high-level operation policy.

## Contents

- `backend.hpp`: common backend include point.
- `blas/`: BLAS-family backend wrappers, vendor detection helpers, OpenBLAS/MKL
  adapters, reference BLAS prototypes, and MPLAPACK binary128 hooks.
- `lapack/`: LAPACK-family declarations grouped by problem shape, with
  reference and MPLAPACK-backed variants.
- `cuda/`: CUDA backend target wiring and placeholders.
- `cusolver/`: cuSOLVER backend target wiring and placeholders.

## Notes

- Prefer explicit backend capability checks over assuming one global vendor
  stack supports every operation and scalar type.
- Backend wrappers sit below `linalg/` and `kernel/`; they should not grow
  front-end tensor policy or symmetry lowering rules.
- Keep external-library warning, option, and include-directory behavior local to
  the dependency target rather than leaking it through Uni20 targets.
