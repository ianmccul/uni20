# Tensor Operations, Kernel Dispatch, and Async Lowering

**Status:** follow-up work identified in the 2026-07-15 design review.

## C++ Preconditions and Async Failure Reporting

- Define the shape and argument-validation policy at each linalg layer. The
  expected native C++ behavior for a violated operation precondition is a
  diagnostic followed by abort; Python bindings must validate user expressions
  before calling such C++ entry points.
- Keep leaf-kernel `CHECK` assertions as protection against broken operation
  wrappers and backend logic. Decide whether Tensor-facing wrappers should use a
  distinct precondition facility so diagnostics identify the operation and the
  incompatible extents before dispatch.
- Extend async failure diagnostics with structured operation context. Observed
  failures already propagate through `Async<T>`; the remaining work is to
  retain the operation name, source location, relevant shape/value information,
  and available task-creation or C++ stack traces.
- Decide how the scheduler should report a failed sinkless bare `AsyncTask`,
  which has no result through which its exception can be observed. Keep the
  established terminal-value policy for `Async<T>`: cancellation is harmless
  absent-value control flow, observed failures propagate, and an unobserved
  failed value is discarded.
- Reconcile synchronous abort semantics with coroutine failure propagation.
  Do not convert ordinary native C++ precondition violations into recoverable
  exceptions merely for consistency with Async or Python.

## Krylov and Kernel Dispatch

- Make Krylov dense operations call the ordinary Tensor linalg interfaces so
  backend selection and kernel diagnostics are exercised by real algorithms.
- Inventory the local operations in `krylov/dense_linalg.hpp`, including matrix
  copy, GEMV, rank-one updates, vector copy/scale/AXPY, inner products, and
  norms. Add operation tags and dispatched kernels where an ordinary Uni20
  operation is missing.
- Move generic accessor-respecting CPU implementations into
  `CpuReferenceBackend` kernels where that produces a reusable operation. Keep
  genuinely Krylov-specific algorithm logic in the Krylov layer.
- Replace the private dense arithmetic inside the CPU matrix exponential with
  ordinary Tensor operations where practical. Add dispatched matrix solve and
  any other missing operations before trying to route the Pade products and
  solve through GEMM/LAPACK.
- Preserve scalar-generic and binary128 behavior while removing the private
  dense-matrix implementation path.

## Async Tensor Views and Materialization

- Use `async::reshape_view` as the reference implementation for future
  structural aliases. It retains the complete parent chain, shares the exact
  `EpochQueue`, defers descriptor resolution until the parent epoch is readable,
  and preserves contiguous order in a canonical static layout type across
  nested semantic views.
- Add synchronous slice descriptors, then expose owner-retaining Async slice
  aliases with the same mutable write-through versus read-only semantic split.
- Extend alias-composition tests whenever a new view adaptor can appear between
  structural views. Never wrap a bare mdspan or borrowed synchronous view in an
  independent Async timeline.
- Add Async materialization operations such as copy or `make_tensor` after the
  alias/view surface is established.
