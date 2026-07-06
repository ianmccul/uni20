# `src/uni20/backend/lapack`

This directory contains raw and checked LAPACK wrappers for configured CPU
providers. It is the external-library boundary below dense linalg policy.

## Contents

- `lapack.hpp`: checked wrappers that translate LAPACK `info` values into C++
  exceptions or status returns.
- `common.hpp`: shared LAPACK backend helpers.
- `reference/`: ordinary LAPACK provider declarations grouped by routine
  family.
- `mplapack/`: MPLAPACK-backed declarations for supported extended-precision
  scalar types.

## Notes

- Keep pointer/leading-dimension interfaces and provider-specific overloads in
  this backend layer.
- Shape validation, view materialization, and tensor allocation policy belong in
  higher linalg helpers.
