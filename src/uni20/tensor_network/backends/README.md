# Tensor-Network Backends

This directory contains operation-specific executors for tensor-network
algorithms. The logical tensor-network plan remains independent of contraction
order and placement; a backend binds that plan to an execution strategy,
intermediate storage, and lower-level dense-kernel selector.

## Contents

- `right_first_rabc.hpp`: right-first R/A/B/C contraction backend and prepared
  executor. It groups repeated `(B, C)` products, allocates reusable
  intermediates in the selected leaf storage, and dispatches the resulting
  dense contractions through its nested selector.

## Layering

These backends operate on symmetry-aware `BlockTensor` views and retain the
logical block keys supplied by the operation plan. Dense block products lower
through the ordinary linalg contraction interface; provider calls and raw CUDA
kernels remain in the lower backend and linalg layers.

## Related Documentation

- [Tensor-network source map](../)
- [R/A/B/C contraction scheduling](../../../../docs/tensor_network/rabc_contraction_scheduling.md)
- [Tensor-network documentation](../../../../docs/tensor_network/)
- [Kernel dispatch](../../../../docs/architecture/kernel_dispatch.md)
