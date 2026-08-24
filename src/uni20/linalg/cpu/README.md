# src/uni20/linalg/cpu

This directory contains ordinary accessor-respecting CPU linalg leaves over
already resolved host mdspans. It is below operation-tag dispatch and host
access acquisition.

## Contents

- `contract.hpp`: arbitrary-rank accessor-respecting pairwise contraction.
- `gemm.hpp`: rank-two reference GEMM compatibility probe and implementation.
- `gemv.hpp`: rank-two-by-rank-one reference GEMV compatibility probe and
  implementation.

## Rules

- Functions in this directory accept resolved `MdspanLike` operands. They do
  not accept Tensor views, select a backend, acquire leases, or redispatch an
  operation tag.
- Operation-tag eligibility, runtime decline, and host lease acquisition belong
  in the [`backends/cpu/`](../backends/cpu/) adapters.
- Access every element through its mdspan accessor. A pointer-shaped data handle
  does not permit bypassing transformed or proxy-reference semantics.
- Keep provider-specific BLAS calls in the [BLAS adapter layer](../blas/).

## Related Documentation

- [Linalg source map](../)
- [Mdspan linear algebra dispatch](../../../../docs/linalg/mdspan_dispatch.md)
- [BLAS/LAPACK mdspan wrappers](../../../../docs/linalg/blas_lapack_wrappers.md)
- [Kernel dispatch](../../../../docs/architecture/kernel_dispatch.md)
