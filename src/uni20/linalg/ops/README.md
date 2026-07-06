# `src/uni20/linalg/ops`

This directory contains dense matrix operation descriptors and public linalg
operation wrappers.

## Contents

- `matrix_ops.hpp`: copy, identity fill, matrix arithmetic, multiplication, and
  dense matrix exponential wrappers over rank-2 tensor views.

## Notes

- Operation wrappers should validate linalg-level shape requirements before
  dispatching to a backend.
- Keep backend-specific implementation details in `../backends`.
