# Linear Algebra Examples

- `blas_example.cpp` prints the configured BLAS vendor and version. Its target
  is available only when the BLAS backend target exists.
- `kernel_dispatch_example.cpp` demonstrates compile-time type probing,
  runtime decline, ordered fallback, and successful dispatch.
- `kernel_dispatch_error_example.cpp` demonstrates structured errors for
  exhausted runtime candidates and type-level rejection in recoverable mode.
- `gemm_dispatch_example.cpp` performs Tensor GEMM, displays operands/results,
  reports backend diagnostics, and supports `fp32`, `fp64`, and configured
  `fp128` precision.

See the [examples index](../), [Kernel Dispatch](../../docs/architecture/kernel_dispatch.md),
and [Linear Algebra](../../docs/linalg/).
