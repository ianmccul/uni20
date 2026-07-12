# `src/uni20/linalg/ops`

This directory contains dense matrix operation descriptors and public linalg
operation wrappers.

## Contents

- `gemm.hpp`: the GEMM operation tag and fixed-output Tensor front end.
- `gemv.hpp`: the GEMV operation tag and fixed-output Tensor front end.

## Notes

- Operation wrappers should validate linalg-level shape requirements before
  dispatching to a backend.
- Bare mdspans call `dispatch_kernel(selector, operation, operands...)`
  directly; do not add operation-specific aliases for generic dispatch.
- Keep backend-specific implementation details in `../backends`.
