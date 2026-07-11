# `src/uni20/linalg/backends`

This directory contains dense linalg backend front ends. These files translate
Uni20 dense matrix operations into a concrete backend family while staying above
the raw external-library wrappers in `src/uni20/backend`.

## Contents

- `cpu/`: CPU dense matrix helpers and CPU matrix exponential implementation.
- `blas/`: operation-tag backend adapters that delegate to mdspan BLAS helpers.
- `lapack/`: LAPACK-backed matrix operation entry points.
- `cusolver/`: cuSOLVER-backed matrix operation entry points.

## Notes

- Backend front ends should own linalg-level policy such as result allocation,
  view preparation, and operation selection.
- Raw provider signatures and ABI details belong in `src/uni20/backend`.
