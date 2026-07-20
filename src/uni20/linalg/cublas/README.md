# src/uni20/linalg/cublas

This directory contains provider-ready cuBLAS operations after Tensor and
mdspan policy has been lowered but before the operation-tag backend adapter.

## Contents

- `gemm.hpp`: checked GEMM over BLAS-compatible matrix descriptors or opaque
  CUDA mdspans. The CUDA-mdspan path validates placement and layout, leases a
  cuBLAS handle and idle stream, opens synchronized buffer access, and calls
  the non-suspending provider wrapper.

## Notes

- Keep raw cuBLAS API calls and handle ownership in
  [`backend/cublas/`](../../backend/cublas/).
- Keep operation-tag acceptance and dispatch adaptation in
  [`linalg/backends/cublas/`](../backends/cublas/).
- Runtime layout rejection must occur before resource acquisition. Once a
  provider call is submitted, failure is terminal rather than a clean backend
  decline.

## Related Documentation

- [Linalg source map](../)
- [CUDA runtime foundation](../../../../docs/backends/cuda/runtime.md)
- [CUDA kernel dispatch and provider scheduling](../../../../docs/backends/cuda/kernel_dispatch.md)
- [Dense BLAS/LAPACK wrapper coverage](../../../../docs/linalg/dense_blas_lapack_coverage.md)
