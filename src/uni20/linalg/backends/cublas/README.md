# src/uni20/linalg/backends/cublas

This directory adapts opaque CUDA mdspan operands to cuBLAS operation tags.
`CublasBackend` validates and stages the mdspans, blocks for an execution lease,
opens synchronized CUDA buffer access, and calls the provider-ready leaf. This
is the ordinary direct Tensor path.

The current first operation is GEMM. Async `assign_product` and `add_product`
await `co_gemm`, which binds a CUDA child to the operand device, awaits the
execution lease, and invokes the same prepared provider leaf without
redispatching through `CublasBackend`.

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
