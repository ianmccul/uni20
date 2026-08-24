# src/uni20/linalg

This directory contains dense linear-algebra front ends and operation
descriptors. It is the policy layer that selects or describes dense operations
before they lower to backend wrappers and kernels.

## Contents

- `linalg.hpp`: public include point for the dense linalg layer.
- `operation_tags.hpp`: central catalogue of dispatchable operation values and
  their diagnostic names, including callable-carrying elementwise operation
  descriptors.
- `elementwise_functions.hpp`: named backend-neutral function objects that may
  have explicit precompiled backend lowerings.
- `reduction_axes.hpp`: normalized reduced/surviving axis descriptors shared by
  reduction front ends and backends.
- `contraction_axes.hpp`: normalized contracted/surviving axis descriptors for
  fixed-rank pairwise contractions.
- `contraction_strides.hpp`: jointly merged M/N/K stride metadata plus direct
  and one-residual-axis rank-two GEMM projections.
- `backend_selector.hpp`: ordered backend selector values, Uni20-owned
  operation/storage defaults, user selector overrides, and the stateless host
  backend entries shared with tensor storage.
- `dispatch.hpp`: operation-value backend-list dispatch helpers.
- `dispatch_diagnostics.hpp`: disabled-by-default structured observation of
  ordered backend walks.
- `kernel_attempt.hpp`, `dispatch_error.hpp`,
  `dispatch_error_presentation.hpp`: clean backend-decline results and terminal
  dispatch-failure diagnostics.
- `async.hpp`: opt-in include point for scheduled `Async<Tensor>` transforms,
  reductions, matrix products, eigensystems, and exact or truncating SVD.
- [`async/`](async/): all-async Tensor wrappers over the synchronous operation layer.
- [`blas/`](blas/): mdspan-to-BLAS-compatible descriptor and direct wrapper helpers.
- [`cublas/`](cublas/): provider-ready cuBLAS operations over staged CUDA
  mdspans and synchronized buffer access.
- [`cpu/`](cpu/): accessor-respecting reference GEMM and GEMV leaves over
  resolved host mdspans.
- [`ops/`](ops/): Tensor-facing dense operation wrappers.
- [`backends/`](backends/): operation-tag backend implementations.
- [`backends/direct_gemm/`](backends/direct_gemm/): direct tensor-contraction lowering through a retained GEMM selector.
- [`backends/looped_gemm/`](backends/looped_gemm/): residual-M/N tensor-contraction loops through a retained GEMM selector.
- [`backends/blas/`](backends/blas/): operation-tag BLAS backend adapters.
- [`backends/cublas/`](backends/cublas/): provider-ready cuBLAS backend adapters.
- [`backends/cpu/`](backends/cpu/): generic CPU operation-tag kernels, dense matrix helpers, and
  the current dense matrix exponential implementation.
- [`backends/lapack/`](backends/lapack/): LAPACK-backed linalg entry points.
- [`backends/cusolver/`](backends/cusolver/): cuSOLVER-backed linalg entry points.

## Notes

- Keep dense linalg APIs separate from matrix-free Krylov APIs in `krylov/`.
- Backend-specific code should stay in `backends/` and call provider wrappers
  under `backend/` where appropriate.
- Backend implementations include `operation_tags.hpp`; do not redeclare
  operation identities locally.
- Bare-mdspan GEMM, GEMV, and copy calls use `dispatch_kernel` with their
  operation tag and an explicit selector. Tensor-view operands use the
  fixed-output `gemm` or `gemv` front end so it can derive their common storage
  selector, or they may supply an explicit selector override.
- Async Tensor operations live in the opt-in `async/` layer. They resolve the
  static storage selector before scheduling, await Tensor values, and then call
  these same synchronous Tensor front ends with that selector; backends do not
  depend on the async runtime.
- Async sum wrappers validate axes before submission, defer shape preparation
  and dispatch until the input is readable, and return either a
  storage-preserving async Tensor or a nonblocking `Async<Element>` host result.
- `copy_op` is the semantic element-copy operation used by Tensor `copy` and
  `make_tensor`. Its CPU backend respects accessors. `CudaReferenceBackend`
  handles canonical contiguous host/device and device/device transfers plus
  registered same-device positive-strided accessor lowerings. Future BLAS
  matrix-copy extensions may accept
  representable rank-two layouts and conjugating views; strict BLAS/LAPACK
  compute wrappers still do not materialize implicitly.
- `transform_op<F>` and `transform_inplace_op<F>` carry a const-invoked
  callable through dispatch. The CPU reference backend supports arbitrary rank
  and input arity. The CUDA reference backend explicitly registers
  same-element-type unary and binary arithmetic function objects plus stateful
  `linalg::scale<Factor>`; other optimized callable/layout combinations belong
  in specialized backends.
- `sum_reduction_op<R, N>` carries normalized reduced and surviving axes.
  Tensor front ends remove the selected axes, preserve canonical result layout,
  and use the generic CPU reference executor when no earlier backend accepts.
- Dense linalg operations use operation values, `kernel_accepts_types`, and
  `try_kernel`; the former backend-tag selector hierarchy has been removed.
- Default selector resolution gives a user `backend_selector_override` complete
  replacement priority over Uni20's `backend_selector_default`; absent either,
  it uses the storage policy's general selector.
- `kernel_type_candidates(...)` filters a selector to its ordered `yes` and
  `maybe` type candidates while preserving backend values, allowing shared
  conformance tests to exercise every compatible backend independently.
- Scalar-generic code should use Uni20 scalar traits and numeric limits from
  `core/`.

## Related Documentation

- [Source tree map](../)
- [Linear algebra documentation](../../../docs/linalg/)
- [Dense BLAS/LAPACK wrapper coverage](../../../docs/linalg/dense_blas_lapack_coverage.md)
- [Kernel dispatch](../../../docs/architecture/kernel_dispatch.md)
- [Tensor operations](../../../docs/tensor/operations.md)
