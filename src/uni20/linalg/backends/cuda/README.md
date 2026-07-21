# CUDA Reference Linalg Backend

This directory contains generic CUDA runtime implementations of linalg
operation tags. `CudaReferenceBackend` follows provider backends such as cuBLAS
in `CudaStorage`'s ordered selector and handles operations that do not require a
provider library.

The initial `copy_op` implementation accepts only unique, exhaustive strided
mdspans with matching physical order and default host or CUDA accessors. It
uses blocking `cudaMemcpy` for pageable host transfers and stream-ordered
device or peer copies for CUDA-to-CUDA transfers. Accessor transforms and
layout conversion require a later CUDA elementwise kernel rather than a raw
byte transfer.

## Related Documentation

- [Linalg backend source map](../)
- [CUDA backend guide](../../../../../docs/backends/cuda/)
- [Tensor scalar and transfer design](../../../../../docs/tensor/scalar_tensors_and_transfer.md)
