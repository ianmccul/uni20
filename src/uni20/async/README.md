# `src/uni20/async`

This directory contains Uni20's coroutine-based async runtime. The runtime owns
task lifetime, dependency ordering, and buffer access discipline; library code
should use these abstractions instead of raw threads or ad hoc synchronization.

## Contents

- `async.hpp`, `async_impl.hpp`, `async_node.hpp`, `async_ops.hpp`: async value
  wrappers and dependency graph plumbing.
- `async_task.hpp`, `async_task_impl.hpp`, `async_task_promise.hpp`,
  `awaiters.hpp`, `cuda_task.hpp`: coroutine task wrappers and await support.
- `scheduler.hpp`, `debug_scheduler.hpp`, `tbb_scheduler.hpp`,
  `tbb_numa_scheduler.hpp`: scheduler interfaces and implementations.
- `epoch_context.hpp`, `epoch_queue.hpp`: epoch ordering and causal execution
  support.
- `buffers.hpp`, `storage_buffer.hpp`, `shared_storage.hpp`,
  `assignment_semantics.hpp`: read/write buffer and storage lifetime helpers.
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
