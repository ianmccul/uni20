# CUDA Reference Linalg Backend

This directory contains generic CUDA runtime implementations of linalg
operation tags. `CudaReferenceBackend` follows provider backends such as cuBLAS
in `CudaStorage`'s ordered selector and handles operations that do not require a
provider library.

The initial `copy_op` implementation accepts only unique, exhaustive strided
mdspans with matching physical order. It uses blocking `cudaMemcpy` for
pageable host transfers and stream-ordered runtime copies for raw device or
peer transfers. Same-device copies with conjugating CUDA accessors instead use
the reference backend's typed elementwise CUDA kernel. The kernel resolves
persistent `uni20::complex<T>` storage through CUDA execution accessors and
publishes its stream completion through the same buffer ledgers as the runtime
copy path. Same-buffer transformed copies still decline until alias-aware
single-write acquisition is implemented.

## Related Documentation

- [Linalg backend source map](../)
- [CUDA backend guide](../../../../../docs/backends/cuda/)
- [Tensor scalar and transfer design](../../../../../docs/tensor/scalar_tensors_and_transfer.md)
