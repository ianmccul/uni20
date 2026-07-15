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
- Investigate whether failures inside scheduled operations should propagate as
  structured exceptions through `Async<T>`. A useful exception should retain
  the operation name, source location, relevant shape/value information, and
  available task-creation or C++ stack traces.
- Preserve the established terminal-value policy: cancellation is harmless
  absent-value control flow; observed failures propagate; an unobserved failed
  `Async<T>` is discarded. A sinkless bare `AsyncTask` has no result through
  which a failure can be observed, so decide how the scheduler should report
  such failures separately.
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

## Async Tensor Views and Aliases

- Add public Async overloads for structural views, beginning with
  `async::reshape_view` and then slices as those synchronous views are added.
- Build these as owner-retaining `Async<TensorView>` aliases that share the
  parent's exact `EpochQueue`; never wrap a bare mdspan or synchronous view in
  an independent Async timeline.
- Preserve the current semantic split: conjugating and const views are
  read-only, while mutable structural views use explicit write-through
  assignment without rebinding their owner or epoch queue.
- Test parent lifetime retention, queue identity, pending writer ordering,
  nested alias chains, write-through assignment, and failure propagation.
- Add Async materialization operations such as copy or `make_tensor` after the
  alias/view surface is established.
