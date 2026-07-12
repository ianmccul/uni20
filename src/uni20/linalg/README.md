# `src/uni20/linalg`

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
- `blas/`: mdspan-to-BLAS-compatible descriptor and direct wrapper helpers.
- `ops/`: operation descriptors such as matrix-operation tags.
- `backends/blas/`: operation-tag BLAS backend adapters.
- `backends/cpu/`: generic CPU operation-tag kernels, dense matrix helpers, and
  the current dense matrix exponential implementation.
- `backends/lapack/`: LAPACK-backed linalg entry points.
- `backends/cusolver/`: cuSOLVER-backed linalg entry points.

## Notes

- Keep dense linalg APIs separate from matrix-free Krylov APIs in `krylov/`.
- Backend-specific code should stay in `backends/` and call through the lower
  `backend/` and `kernel/` layers where appropriate.
- Backend implementations include `operation_tags.hpp`; do not redeclare
  operation identities locally.
- Bare-mdspan GEMM and GEMV calls use `dispatch_kernel` with their operation tag
  and an explicit selector. Tensor-view operands use the fixed-output `gemm` or
  `gemv` front end so it can derive their common storage selector, or they may
  supply an explicit selector override.
- New dense linalg operations use operation tags, `kernel_accepts_types`, and
  `try_kernel`. The former `cpu_tag`/`lapack_tag` matrix-operation front end has
  been removed; those tags remain only in older, separate kernel lineages that
  have not yet migrated to operation-tag dispatch.
- Scalar-generic code should use Uni20 scalar traits and numeric limits from
  `core/`.
