# src/uni20/linalg/backends/cublas

This directory adapts opaque CUDA mdspan operands to cuBLAS operation tags.
`CublasBackend` validates and stages the mdspans before acquiring resources,
then obtains the cuBLAS execution pool owned by `DeviceResources`, leases a
handle and idle stream, opens synchronized CUDA buffer access, and calls the
provider-ready leaf.

The current first operation is GEMM. Ordinary Tensor code reaches it through
`linalg::gemm(output, alpha, lhs, rhs, beta)`; the Tensor front end resolves
mdspans and dispatches the storage-selected `CublasBackend`. The direct path
uses blocking resource admission. Future async CUDA lowering will await the
same execution resources before entering the non-suspending provider leaf.

An empty GEMM output succeeds before operand staging. cuBLAS accepts a zero
inner extent with null zero-sized input buffers and applies the degenerate
`C = beta*C` operation, so canonical CUDA matrices retain provider execution for
that case. A future `CudaReferenceBackend` after `CublasBackend` in the
storage-selected backend list will provide complete CUDA-domain handling for
layouts and accessors that cuBLAS cannot represent.

The Tensor conformance tests share their scalar and canonical-layout cases with
the host GEMM backends. cuBLAS-specific tests cover opaque buffer offsets,
transpose and conjugate-transpose subviews, padded leading dimensions, clean
layout decline before resource acquisition, and hard device/alias contract
violations. Unknown accessor semantics are rejected at type probing rather than
bypassed through the opaque handle.

## Related Documentation

- [Linalg backend source map](../)
- [Provider-ready cuBLAS operations](../../cublas/)
- [CUDA kernel dispatch and provider scheduling](../../../../../docs/backends/cuda/kernel_dispatch.md)
- [Dense BLAS/LAPACK wrapper coverage](../../../../../docs/linalg/dense_blas_lapack_coverage.md)
