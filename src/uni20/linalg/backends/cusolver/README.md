# `src/uni20/linalg/backends/cusolver`

This directory is reserved for cuSOLVER operation-tag backend adapters.

## Notes

- Gate operations through `kernel_accepts_types` and `try_kernel` before
  committing to a cuSOLVER path.
- Keep CUDA library ABI details in `src/uni20/backend/cusolver`.
