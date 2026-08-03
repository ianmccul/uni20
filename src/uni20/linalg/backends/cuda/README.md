# CUDA Reference Linalg Backend

This directory contains generic CUDA runtime implementations of linalg
operation tags. `CudaReferenceBackend` follows provider backends such as cuBLAS
in `CudaStorage`'s ordered selector and handles operations that do not require a
provider library.

The `copy_op` implementation keeps runtime-copy fast paths for compatible raw
contiguous mappings. Same-device positive-strided mappings whose compact rank is
at most eight, including padding and differing physical order, use the reference
backend's typed elementwise CUDA kernel. `elementwise_plan.hpp` lowers the same
backend-neutral affine plan used by the CPU executor to a device POD, preferring
32-bit logical indices and offsets when every reachable offset fits and using a
64-bit payload otherwise. The compiled executor is arity-generic and separates
overwrite operations, which do not read the output, from update operations,
which do.

Copy is the first one-input overwrite instantiation. `conjugate_inplace_op` is
the first zero-input update instantiation: it reads and writes each output
element through one exclusive CUDA access. Real and integer conjugation succeeds
without acquiring a stream or launching because it is semantically the identity.
Complex conjugation supports persistent `uni20::cfloat` and `uni20::cdouble`
storage through CUDA execution proxies.

The kernels publish stream completion through the same buffer ledgers as the
runtime copy path. Nonpositive strides on active axes cleanly decline for
nontrivial elementwise work; supporting a future negative-stride Uni20 layout
requires descriptor rebasing and signed traversal rather than raw runtime copying.
Distinct-offset views into one CUDA buffer use a single exclusive access and
rely on the C++ copy precondition that the operands do not destructively
overlap. Nontrivial same-offset transformations decline.

## Related Documentation

- [Linalg backend source map](../)
- [CUDA backend guide](../../../../../docs/backends/cuda/)
- [Tensor scalar and transfer design](../../../../../docs/tensor/scalar_tensors_and_transfer.md)
