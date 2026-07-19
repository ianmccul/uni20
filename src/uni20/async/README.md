# src/uni20/async

This directory contains Uni20's coroutine-based async runtime. The runtime owns
task lifetime, dependency ordering, and buffer access discipline; library code
should use these abstractions instead of raw threads or ad hoc synchronization.

## Contents

- `async.hpp`, `async_impl.hpp`, `async_node.hpp`, `async_ops.hpp`,
  `async_traits.hpp`: async value wrappers, payload capabilities, and dependency
  graph plumbing.
- `async_errors.hpp`: async cancellation and deadlock exception types.
- `async_task.hpp`, `async_task_impl.hpp`, `async_task_promise.hpp`,
  `awaiters.hpp`: implemented coroutine task wrappers and await support.
- `cuda_task.hpp`: CUDA-specific initial-admission task type and scheduler
  interface. `CudaTask` shares the canonical promise and internal task
  representation with `AsyncTask`, so suspended tasks use the same tested
  continuation and rescheduling machinery.
- `debug_cuda_scheduler.hpp`: deterministic calling-thread CUDA scheduler bound
  to one validated device. It establishes and restores the current CUDA device
  around each run.
- `tbb_cuda_scheduler.hpp`: oneTBB CUDA scheduler with one arena per scheduler
  and an arena observer that establishes and restores the bound device for every
  participating worker or application thread.
- `tbb_task_submission.hpp`: internal task-group registration and non-blocking
  arena-admission helper shared by host and CUDA TBB schedulers.
- `scheduler.hpp`, `debug_scheduler.hpp`, `tbb_scheduler.hpp`,
  `tbb_numa_scheduler.hpp`: scheduler interfaces and implementations.
- `epoch_context.hpp`, `epoch_queue.hpp`: epoch ordering and causal execution
  support.
- `buffers.hpp`, `storage_buffer.hpp`, `shared_storage.hpp`: read and
  exclusive-mutable capabilities, their await paths, and storage lifetime
  helpers.
- `var.hpp`, `future_value.hpp`, `reverse_value.hpp`: async dataflow values.
- `task_registry_*`: optional debug task registry support.
- `async_toys.hpp`, `var_toys.hpp`: experiments and examples; do not treat
  these as stable API.

## Notes

- Coroutine lambdas that return Uni20 async tasks must be captureless and use
  the C++23 `static` lambda modifier. See `/AGENTS.md` for the rule and
  rationale.
- Shared data should flow through the buffer/value abstractions in this module
  so scheduler ordering remains explicit.
- `is_async_alias_v<T>` controls independent-value versus shared-alias identity.
  Assignment to an alias is available only through an ADL-visible
  `assign_through` operation. `WriteBuffer<T>` remains the single exclusive
  epoch capability, with proxy operations constrained by the payload.

## Related Documentation

- [Source tree map](../)
- [Async documentation index](../../../docs/async/)
- [Async storage and identity](../../../docs/async/storage.md)
- [Exceptions and cancellation](../../../docs/async/exceptions_and_cancellation.md)
- [Scheduler routing and task domains](../../../docs/async/scheduler_migration.md)
- [Task registry diagnostics](../../../docs/async/task_registry_debug.md)
