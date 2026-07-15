# Async Quick Reference

This page is intentionally compact. Use it for fast lookup during implementation,
code review, and AI-agent prompting.

For explanations, see:

- `getting_started.md`
- `buffers_and_awaiters.md`
- `coroutines_primer.md`
- `cookbook.md`
- `runtime_model.md`
- `exceptions_and_cancellation.md`

## Core Types

| Type | Purpose | Key methods |
|---|---|---|
| `Async<T>` | Async value + epoch queue | `read()`, `write()`, `wait()`, `get_wait()`, `move_from_wait()` |
| `ReadBuffer<T>` | Read gate for one epoch | `co_await reader`, `transfer()`, `maybe()`, `or_cancel()`, `wait()`, `release()` |
| `WriteBuffer<T>` | Exclusive mutable gate for one epoch | `co_await writer`, `transfer()`, `storage()`, `take()`, `take_release()`, `release()` |
| `AsyncTask` | Move-only coroutine handle owner | schedule via `schedule(...)` |
| `IScheduler` | Scheduler interface | `schedule`, `pause`, `resume`, wait hooks |

Buffer model:

- `ReadBuffer<T>` is shared read access to one epoch
- `WriteBuffer<T>` is exclusive mutable access, including reading the old value
- `maybe()`, `or_cancel()`, `storage()`, `take()`, and `transfer()` select await paths; they do not create new buffers or epochs
- for one queue, an ordinary operation uses any number of readers or one writer, never a separate reader plus writer
- mutation reads the existing value through its one `WriteBuffer<T>`

Async aliases:

- `make_async_alias<View>(parent, args...)` constructs a durable view descriptor
  that retains `parent` and shares its exact epoch queue
- `make_async_alias_from_parent<View>(parent, args...)` also passes the parent's
  stable reserved address to a descriptor that resolves only after the shared
  epoch is readable; `async::reshape_view` uses this form
- alias descriptor types declare `async_alias_tag`, making `Async<View>` copies
  structural handle copies
- ordinary `Async<T>` copies remain scheduled value copies

## Canonical Coroutine Pattern

```cpp
auto kernel = [](ReadBuffer<int> in, WriteBuffer<int> out) static -> AsyncTask {
  auto owned = co_await in.transfer();
  int v = owned.get();
  owned.release();
  co_await out = v + 1;
  co_return;
};
```

Rules:

- coroutine lambdas are captureless
- coroutine lambdas are `static`
- pass buffers as coroutine parameters

## Read/Write Semantics

### Async<T> initialization

- `Async<T>()`: storage exists, value is unconstructed
- `Async<T>(args...)`: value is constructed immediately

### Async<T> assignment

| Expression | Meaning |
|---|---|
| `owning = value` | Rebind: detach and create fresh storage plus a fresh `EpochQueue`. |
| `mutable_alias = value_or_alias` | Write through: retain descriptor, owner, and queue, including for an exact-type source. |
| `read_only_alias = anything` | Ill-formed: assignment to const or transforming read-only proxies is forbidden. |
| `async_assign(destination, source)` | Enroll assignment in the destination's current queue. |

`is_async_alias_v<T>` distinguishes aliases from independent values. Mutable
aliases provide an ADL-visible `assign_through(target, source)` operation;
read-only aliases do not. Every `Async<T>` can produce the same
`WriteBuffer<T>` epoch handle, but its proxy exposes only operations supported
by the payload: alias descriptors are read-only and cannot be emplaced, moved
out, or retargeted.

### ReadBuffer<T> await results

| Expression | Result |
|---|---|
| `co_await reader` (lvalue) | `T const&` |
| `co_await reader.transfer()` | `OwningReadAccessProxy<T>` |
| `co_await reader.maybe()` | `T const*` |
| `co_await reader.transfer().maybe()` | `std::optional<OwningReadAccessProxy<T>>` |
| `co_await reader.or_cancel()` | `T const&` or `task_cancelled` |
| `co_await reader.transfer().or_cancel()` | `OwningReadAccessProxy<T>` or `task_cancelled` |

### WriteBuffer<T> await results

| Expression | Result |
|---|---|
| `co_await writer` | `WriteAccessProxy<T>` (`T&` for values, `T const&` for aliases) |
| `co_await writer.transfer()` | `OwningWriteAccessProxy<T>` |
| `co_await writer.storage()` | `shared_storage<T>&` for independent values only |
| `co_await writer.transfer().storage()` | `OwningStorageAccessProxy<T>` for independent values only |
| `co_await writer.take()` | `T` moved out, then storage destroyed; values only |
| `co_await writer.transfer().take()` | `T` moved out, then storage destroyed and writer released; values only |
| `co_await writer.take_release()` | `T` moved out, then storage destroyed and writer released; values only |
| `co_await writer.transfer().take_release()` | `T` moved out, then storage destroyed and writer released; values only |

Awaiter/proxy distinction:

- expressions such as `writer.storage()` and `writer.take()` create temporary awaiters
- lvalue await paths return borrowed access tied to the original buffer
- `transfer()` moves the epoch capability through the awaiter into an owning result
- repeated awaits on one buffer use the same epoch; they do not enroll another writer

Critical write rule:

- `co_await writer = value` constructs empty storage and otherwise evaluates
  `stored_value = value`
- for aliases, the same expression invokes matching `assign_through` and never
  replaces the descriptor
- proxy assignment requires the source to both construct `T` and make
  `stored_value = source` valid for independent values
- `co_await writer += x` and `co_await writer -= x` also emplace independent values when unconstructed
- value-proxy `get()`, `operator->`, or conversion to `T&` throws `buffer_write_uninitialized` before initialization
- alias-proxy `get()` and `operator->` expose only the const descriptor
- use one writer, without a separate reader, to inspect and mutate an existing value

Explicit alternatives:

- `proxy.emplace(...)` explicitly reconstructs an independent value in the same timeline
- `proxy.get() = value` assigns an independent value known to be constructed
- tensor element copying and backend dispatch use named operations such as `copy(...)`

## Async Ops (async_ops.hpp)

| Helper | Typical use |
|---|---|
| `async_assign(dst, src)` | copy-like value propagation |
| `async_move(dst, src)` | move-like value transfer |
| `async_binary_op(out, a, b, op)` | schedule `out = op(a, b)` |
| `async_compound_op(lhs, rhs, op)` | schedule in-place-style update to async lhs |

Helper awaiters:

- `all(a, b, ...)`
- `try_await(awaitable)`
- `write_to(writer.transfer(), value)`

## Exceptions and Cancellation

Main exception types (`async_errors.hpp`):

- `task_cancelled`
- `buffer_read_uninitialized`
- `buffer_write_uninitialized`
- `async_value_uninitialized`

Unhandled exception behavior:

- promise captures `std::exception_ptr`
- forwards exception to registered sink epochs
- downstream awaiters rethrow on access

Sink registration:

- automatic for `WriteBuffer` coroutine parameters
- explicit via `co_await propagate_exceptions_to(buf1, buf2, ...)`

## Scheduler Choice

| Scheduler | When to use |
|---|---|
| `DebugScheduler` | deterministic tests and semantics debugging |
| `TbbScheduler` | general parallel execution |
| `TbbNumaScheduler` | NUMA-aware execution |

## TaskRegistry Debugging

### Build-time

- `-DUNI20_DEBUG_ASYNC_TASKS=ON`
- `-DUNI20_DEBUG_DAG=ON`
- `-DUNI20_ENABLE_STACKTRACE=ON` when supported; this is the fresh-build default

If `<stacktrace>` is unavailable, the option defaults to `OFF` and the build
continues with degraded stacktrace output. Use `-DUNI20_ENABLE_STACKTRACE=OFF`
to disable otherwise available support explicitly.
`UNI20_DEBUG_DAG` adds Graphviz DOT snapshots and enables task-registry instrumentation.

### Runtime verbosity

Environment variable: `UNI20_DEBUG_ASYNC_TASKS`

- `none`, `off`, `false`, `0` -> no dump
- `basic`, `on`, `true`, `1` -> basic dump
- `full`, `all`, `verbose`, `2` -> full dump

Unknown/unset currently defaults to `basic`.

### DAG snapshots

Use `TaskRegistry::graphviz_dot()` or `TaskRegistry::dump_graphviz()` to capture the
current task/epoch/value graph. With `UNI20_DEBUG_DAG=ON`, coroutine
`ReadBuffer`/`WriteBuffer` parameters become coarse dependency edges before the task
runs, and concrete buffer `co_await` sites add finer dependency edges.
With stacktrace support, DOT labels also include best-effort `created_at`,
`scheduled_at`, and `awaiting_at` source locations, with full traces in tooltips.
Without stacktrace support, the same DAG structure is emitted without provenance.
Use `TaskRegistry::set_stacktrace_options(...)` to control how many frames are
serialized; `max_frames=0` keeps source-location labels but omits full trace text.

Use `TaskRegistry::snapshot()` for structured records, then
`TaskRegistry::diagnose_snapshot(snapshot)` for blocked-task, missing-writer, and
cycle annotations. `TaskRegistry::graphviz_dot(snapshot, diagnostics)` renders the
same captured point in time as DOT without recapturing runtime state.

DOT snapshots also annotate common debug cases:

- blocked readers/writers
- missing writers for currently blocked reads
- dependency cycles inferred from blocked-read and producer edges

Optional diagnostic labels:

```cpp
value.debug_name("x");
task.debug_name("update x");
```

Labels only affect debug output; they do not affect scheduling or dependencies.

For deadlock or external-debug paths, prefer
`TaskRegistry::graphviz_dot_best_effort()` or
`TaskRegistry::dump_graphviz_file_best_effort(path)`. These avoid waiting forever
if a debug lock is held and mark unavailable parts of the graph in the DOT output.

Queued dump request flow:

```cpp
uni20::TaskRegistry::request_graphviz_dump();
uni20::TaskRegistry::service_debug_requests();
```

Schedulers service queued requests at progress points. For out-of-band requests,
start the optional diagnostics service and trigger it with a request file or Linux
signal:

```cpp
uni20::TaskRegistry::DiagnosticsServiceOptions options;
options.request_file = "/tmp/uni20-dag/request";
options.signal_number = SIGUSR1;
uni20::TaskRegistry::start_diagnostics_service(options);
```

Alternatively, setting `UNI20_DEBUG_DAG_SIGNAL=SIGUSR1` before launching the
program starts the service during program initialization. Uni20 blocks and
consumes the signal with `sigtimedwait`; it does not run diagnostics from an
asynchronous signal handler.

Useful environment defaults:

- `UNI20_DEBUG_DAG_OUTPUT_DIR`
- `UNI20_DEBUG_DAG_FILE_PREFIX`
- `UNI20_DEBUG_DAG_REQUEST_FILE`
- `UNI20_DEBUG_DAG_SIGNAL`
- `UNI20_DEBUG_DAG_POLL_MS`
- `UNI20_DEBUG_DAG_STACKTRACE_FRAMES`
- `UNI20_DEBUG_DAG_STACKTRACE_INTERNAL_FRAMES`

## Fast Troubleshooting

| Symptom | Likely cause | First check |
|---|---|---|
| `buffer_write_uninitialized` | requested mutable reference before construction | replace with proxy assignment (`co_await writer = value`) or proxy `emplace(...)` |
| reader blocks forever | read/write dependency cycle or missing release | verify epoch ordering and `reader.release()` placement |
| cancellation unexpectedly propagates | `or_cancel()` path used and no value present | inspect upstream writer and exception sink routing |
| deadlock dump at shutdown | unresolved suspended tasks | inspect `TaskRegistry::dump()` task/epoch links |
| async DAG is unclear | hidden read/write ordering | inspect `TaskRegistry::graphviz_dot()` |

## Where to Look (Code and Tests)

Core code:

- `src/uni20/async/async.hpp`
- `src/uni20/async/buffers.hpp`
- `src/uni20/async/epoch_context.hpp`
- `src/uni20/async/async_task_promise.hpp`
- `src/uni20/async/debug_scheduler.hpp`
- `src/uni20/async/tbb_scheduler.hpp`

Behavioral ground truth:

- `tests/async/test_async_basic.cpp`
- `tests/async/test_async_awaiters.cpp`
- `tests/async/test_task_registry.cpp`
- `tests/async/test_tbb_scheduler.cpp`
- `tests/async/test_reverse_value.cpp`
