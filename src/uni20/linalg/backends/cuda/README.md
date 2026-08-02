# CUDA Reference Linalg Backend

This directory contains generic CUDA runtime implementations of linalg
operation tags. `CudaReferenceBackend` follows provider backends such as cuBLAS
in `CudaStorage`'s ordered selector and handles operations that do not require a
provider library.

The `copy_op` implementation keeps runtime-copy fast paths for compatible raw
contiguous mappings. Same-device positive-strided mappings through rank eight,
including padding and differing physical order, use the reference backend's
typed elementwise CUDA kernel. The kernel resolves persistent
`uni20::complex<T>` storage through CUDA execution accessors and publishes its
stream completion through the same buffer ledgers as the runtime copy path.
Distinct-offset views into one CUDA buffer use a single exclusive access and
rely on the C++ copy precondition that the operands do not destructively
overlap. Nontrivial same-offset transformations decline.

## Related Documentation

- [Linalg backend source map](../)
- [CUDA backend guide](../../../../../docs/backends/cuda/)
- [Tensor scalar and transfer design](../../../../../docs/tensor/scalar_tensors_and_transfer.md)
