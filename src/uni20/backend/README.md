# src/uni20/backend

This directory contains low-level wrappers for external backend libraries. These
headers should expose library capabilities and calling conventions without
owning tensor semantics, symmetry metadata, or high-level operation policy.

## Contents

- `backend.hpp`: common backend include point.
- [`blas/`](blas/README.md): BLAS-family backend wrappers, vendor detection helpers, OpenBLAS/MKL
  adapters, reference BLAS prototypes, and MPLAPACK binary128 hooks.
- [`lapack/`](lapack/README.md): LAPACK-family declarations grouped by problem shape, with
  reference and MPLAPACK-backed variants.
- [`cuda/`](cuda/README.md): CUDA backend target wiring and placeholders.
- [`cusolver/`](cusolver/README.md): cuSOLVER backend target wiring and placeholders.

## Notes

- Prefer explicit backend capability checks over assuming one global vendor
  stack supports every operation and scalar type.
- Backend wrappers sit below `linalg/` and `kernel/`; they should not grow
  front-end tensor policy or symmetry lowering rules.
- Keep external-library warning, option, and include-directory behavior local to
  the dependency target rather than leaking it through Uni20 targets.

## Related Documentation

- [Source tree map](../README.md)
- [Backend documentation](../../../docs/backends/README.md)
- [Kernel dispatch](../../../docs/architecture/kernel_dispatch.md)
- [Linear algebra documentation](../../../docs/linalg/README.md)
