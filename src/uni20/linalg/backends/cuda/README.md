# CUDA Reference Linalg Backend

This directory contains generic CUDA runtime implementations of linalg
operation tags. `CudaReferenceBackend` follows provider backends such as cuBLAS
in `CudaStorage`'s ordered selector and handles operations that do not require a
provider library.

The initial `copy_op` implementation accepts only unique, exhaustive strided
mdspans with matching physical order. It uses blocking `cudaMemcpy` for
pageable host transfers and stream-ordered device or peer copies for
CUDA-to-CUDA transfers. Raw copies require default host or CUDA pointer
accessors. Conjugating inputs are explicitly lowered while staging pageable
host transfers; conjugating CUDA-to-CUDA copies and other accessor transforms
are declined unless materialized first or implemented by a later CUDA
elementwise kernel.

## Related Documentation

- [Linalg backend source map](../)
- [CUDA backend guide](../../../../../docs/backends/cuda/)
- [Tensor scalar and transfer design](../../../../../docs/tensor/scalar_tensors_and_transfer.md)
