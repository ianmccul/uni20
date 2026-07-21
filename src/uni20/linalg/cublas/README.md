# src/uni20/linalg/cublas

This directory contains provider-ready cuBLAS operations after Tensor and
mdspan policy has been lowered but before the operation-tag backend adapter.

## Contents

- `gemm.hpp`: checked GEMM over provider-ready BLAS-compatible matrix
  descriptors and an already acquired cuBLAS execution lease.

## Notes

- Keep raw cuBLAS API calls and handle ownership in
  [`backend/cublas/`](../../backend/cublas/).
- Keep operation-tag acceptance and dispatch adaptation in
  [`linalg/backends/cublas/`](../backends/cublas/).
- Keep CUDA-mdspan recognition, storage-policy lowering, completion-ledger
  access, and blocking/coroutine resource admission in the backend adapter's
  private `detail/` implementation.
- Runtime layout rejection must occur before synchronized buffer access and
  provider submission. Once a provider call is submitted, failure is terminal
  rather than a clean backend decline.

## Related Documentation

- [Linalg source map](../)
- [CUDA runtime foundation](../../../../docs/backends/cuda/runtime.md)
- [CUDA kernel dispatch and provider scheduling](../../../../docs/backends/cuda/kernel_dispatch.md)
- [Dense BLAS/LAPACK wrapper coverage](../../../../docs/linalg/dense_blas_lapack_coverage.md)
