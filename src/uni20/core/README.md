# `src/uni20/core`

This directory contains the smallest Uni20 type and scalar foundations. It is
intended to be cheap to include from any module and should not depend on higher
layers such as tensor, linalg, backend, async, or symmetry.

## Contents

- `types.hpp`: project scalar aliases such as `uni20::complex<T>`.
- `scalar_concepts.hpp`, `scalar_traits.hpp`, `scalar_io.hpp`: scalar
  classification, promotion, and formatting support.
- `scalar_precision.hpp`: provider-neutral runtime selection of configured real
  scalar precisions.
- `numeric_limits.hpp`: Uni20 numeric-limits customization point for generic
  scalar algorithms.
- `math.hpp`: small scalar math helpers.
- `buildinfo.hpp.in`: generated build/environment metadata.
- `dummy.*`: minimal target source used to keep build targets well-formed where
  needed.

## Notes

- Spell complex scalar types as `uni20::complex<T>` in Uni20 code.
- Use `uni20::numeric_limits<T>` in scalar-generic algorithms so non-standard
  scalar types have one project-level customization point.
