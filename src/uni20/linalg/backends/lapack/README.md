# `src/uni20/linalg/backends/lapack`

This directory contains LAPACK-backed dense linalg front-end declarations.

## Contents

- `linalg_lapack.hpp`: LAPACK linalg include point.
- `matrix_ops_lapack.hpp`: LAPACK matrix operation entry points.

## Notes

- This layer should prepare views and select LAPACK operations, then call the
  raw wrappers under `src/uni20/backend/lapack`.
- Keep copy/materialization behavior explicit when adapting tensor views to
  LAPACK-compatible storage.
