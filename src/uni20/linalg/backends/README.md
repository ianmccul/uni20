# src/uni20/linalg/backends

This directory contains dense linalg backend front ends. These files translate
Uni20 dense matrix operations into a concrete backend family while staying above
the [raw external-library wrappers](../../backend/README.md).

## Contents

- [`cpu/`](cpu/README.md): CPU dense matrix helpers and CPU matrix exponential implementation.
- [`blas/`](blas/README.md): operation-tag backend adapters that delegate to mdspan BLAS helpers.
- [`lapack/`](lapack/README.md): LAPACK-backed matrix operation entry points.
- [`cusolver/`](cusolver/README.md): cuSOLVER-backed matrix operation entry points.

## Notes

- Backend front ends should own linalg-level policy such as result allocation,
  view preparation, and operation selection.
- Raw provider signatures and ABI details belong in the
  [backend source layer](../../backend/README.md).

## Related Documentation

- [Linalg source map](../README.md)
- [Linear algebra documentation](../../../../docs/linalg/README.md)
- [Kernel dispatch](../../../../docs/architecture/kernel_dispatch.md)
