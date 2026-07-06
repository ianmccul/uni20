# `src/uni20/linalg/backends/cusolver`

This directory contains cuSOLVER-backed dense linalg front-end declarations.

## Contents

- `linalg_cusolver.hpp`: cuSOLVER linalg include point.
- `matrix_ops_cusolver.hpp`: cuSOLVER matrix operation entry points.

## Notes

- Gate operations through backend capability checks before committing to a
  cuSOLVER path.
- Keep CUDA library ABI details in `src/uni20/backend/cusolver`.
