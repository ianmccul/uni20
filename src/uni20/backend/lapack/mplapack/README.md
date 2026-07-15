# src/uni20/backend/lapack/mplapack

This directory contains MPLAPACK-backed LAPACK declarations for scalar types
that are not covered by the ordinary LAPACK provider.

## Contents

- `common.hpp`: shared MPLAPACK declaration helpers.
- `band.hpp`, `general.hpp`, `norms.hpp`, `tridiagonal.hpp`: routine-family
  declarations.

## Notes

- Keep MPLAPACK-specific signatures and configuration checks isolated here.
- Generic callers should include `../lapack.hpp` unless they are adding or
  repairing provider-specific wrappers.

## Related Documentation

- [LAPACK provider source layer](../)
- [MPLAPACK binary128](../../../../../docs/linalg/mplapack_binary128.md)
