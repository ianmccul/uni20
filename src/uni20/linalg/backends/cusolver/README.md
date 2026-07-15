# src/uni20/linalg/backends/cusolver

This directory is reserved for cuSOLVER operation-tag backend adapters.

## Notes

- Gate operations through `kernel_accepts_types` and `try_kernel` before
  committing to a cuSOLVER path.
- Keep CUDA library ABI details in the
  [cuSOLVER provider layer](../../../backend/cusolver/).

## Related Documentation

- [Linalg backend source map](../)
- [cuSOLVER architecture](../../../../../docs/backends/cuda/cusolver.md)
- [CUDA runtime resolution](../../../../../docs/backends/cuda/runtime_resolution.md)
