# src/uni20/linalg/backends/cpu/detail

This directory contains implementation utilities shared by CPU linalg
backends.

## Contents

- `compensated_sum.hpp`: same-precision Neumaier accumulators for real and
  complex reduction components.

Public CPU backend kernels live in the [parent directory](../).

See [Tensor Operations](../../../../../../docs/tensor/operations.md) for the
same-precision reduction and scalar-result policy.
