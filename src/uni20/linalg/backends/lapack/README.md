# `src/uni20/linalg/backends/lapack`

This directory is reserved for LAPACK operation-tag backend adapters.

## Notes

- Each operation should provide `kernel_accepts_types(LapackBackend, ...)` and
  `try_kernel(LapackBackend, ...)`, then call the raw wrappers under
  `src/uni20/backend/lapack`.
- Keep copy/materialization behavior explicit when adapting tensor views to
  LAPACK-compatible storage.
