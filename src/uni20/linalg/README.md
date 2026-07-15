# src/uni20/linalg

This directory contains dense linear-algebra front ends and operation
descriptors. It is the policy layer that selects or describes dense operations
before they lower to backend wrappers and kernels.

## Contents

- `linalg.hpp`: public include point for the dense linalg layer.
- `operation_tags.hpp`: central catalogue of dispatchable operation values and
  their diagnostic names.
- `backend_selector.hpp`: ordered backend selector values and the stateless
  host backend entries shared with tensor storage.
- `dispatch.hpp`: operation-tag backend-list dispatch helpers.
- `dispatch_diagnostics.hpp`: disabled-by-default structured observation of
  ordered backend walks.
- `kernel_attempt.hpp`, `dispatch_error.hpp`,
  `dispatch_error_presentation.hpp`: clean backend-decline results and terminal
  dispatch-failure diagnostics.
- `async.hpp`: opt-in include point for scheduled `Async<Tensor>` operations.
- [`async/`](async/README.md): all-async Tensor wrappers over the synchronous operation layer.
- [`blas/`](blas/README.md): mdspan-to-BLAS-compatible descriptor and direct wrapper helpers.
- [`ops/`](ops/README.md): Tensor-facing dense operation wrappers.
- [`backends/`](backends/README.md): operation-tag backend implementations.
- [`backends/blas/`](backends/blas/README.md): operation-tag BLAS backend adapters.
- [`backends/cpu/`](backends/cpu/README.md): generic CPU operation-tag kernels, dense matrix helpers, and
  the current dense matrix exponential implementation.
- [`backends/lapack/`](backends/lapack/README.md): LAPACK-backed linalg entry points.
- [`backends/cusolver/`](backends/cusolver/README.md): cuSOLVER-backed linalg entry points.

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
- `copy_op` is the semantic element-copy operation used by Tensor `copy` and
  `make_tensor`. Its CPU backend respects accessors. Future BLAS matrix-copy
  extensions may accept representable rank-two layouts and conjugating views;
  strict BLAS/LAPACK compute wrappers still do not materialize implicitly.
- Dense linalg operations use operation tags, `kernel_accepts_types`, and
  `try_kernel`; the former backend-tag selector hierarchy has been removed.
- Scalar-generic code should use Uni20 scalar traits and numeric limits from
  `core/`.

## Related Documentation

- [Source tree map](../README.md)
- [Linear algebra documentation](../../../docs/linalg/README.md)
- [Dense BLAS/LAPACK wrapper coverage](../../../docs/linalg/dense_blas_lapack_coverage.md)
- [Kernel dispatch](../../../docs/architecture/kernel_dispatch.md)
- [Tensor operations](../../../docs/tensor/operations.md)
