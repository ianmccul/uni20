# Uni20 Async Runtime: AI Guidance

- **Audience:** remote assistants, coding agents, and reviewers
- **Authority:** non-normative retrieval summary
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-18
- **Canonical sources:** `docs/async/runtime_model.md`,
  `docs/async/buffers_and_awaiters.md`, `docs/async/storage.md`,
  `src/uni20/async/`, and `tests/async/`

## Hard invariants

- Async correctness comes from `EpochQueue` ordering, not scheduler timing.
- Coroutine lambdas returning Uni20 async task types must be captureless and `static`.
- Values needed by a coroutine must be passed as parameters.
- One ordinary operation uses either readers or one writer for each queue.
- A `WriteBuffer<T>` is exclusive mutable access, not write-only access.
- `Async<T>()` has valid storage but no constructed `T`.
- `Async<T>(args...)` has a constructed, immediately readable `T`.
- `Async` does not prove alias safety across distinct objects.
- Parent/alias queue enrollment must be externally serialized.

## Core model

### `Async<T>`

- Owns shared storage and an `EpochQueue` handle.
- `read()` enrolls a `ReadBuffer<T>` in one epoch.
- `write()` enrolls a `WriteBuffer<T>` in the next write epoch.
- Buffers retain storage and their selected epoch even if the originating
  `Async<T>` is moved or destroyed.

### `EpochQueue`

- Represents one causal timeline.
- Conceptual order is `writer_n -> readers_n -> writer_{n+1}`.
- Scheduler implementations decide when ready work runs; they do not define legality.
- Calls that add buffers to the same queue must be externally serialized.
- Whole-owner queue sharing may conservatively serialize disjoint aliases.
  Finer granularity requires explicit subrange hazard tracking.

### `ReadBuffer<T>`

- Shared read capability for one selected epoch.
- `co_await reader` returns borrowed `T const&`.
- `co_await reader.transfer()` returns an owning read proxy.
- Borrowed access is tied to the `ReadBuffer` lifetime.
- `or_cancel()` surfaces cancellation as `task_cancelled`.
- Release a reader early when a later conflicting epoch must become runnable.

### `WriteBuffer<T>`

- Move-only exclusive mutable capability for one write epoch.
- `co_await writer` returns a borrowed write proxy.
- `co_await writer.transfer()` returns an owning write proxy.
- The writer may inspect/mutate an existing value, construct absent storage,
  replace it, or move it out, subject to payload capability.
- Ordinary mutation uses one writer and no separate reader on the same queue.
- `co_await writer = value` constructs empty value storage and otherwise evaluates
  the stored type's assignment expression.
- `emplace(...)` reconstructs inside the existing timeline; it is not an async rebind.
- Recommend `return co_await writer.transfer()` rather than returning a borrowed proxy.

## Await paths are not capabilities

`maybe()`, `or_cancel()`, `storage()`, `take()`, `take_release()`, and
`transfer()` adapt one existing buffer. They do not create another buffer,
epoch, reader, or writer.

## Value versus alias assignment

### Independent values

- Copy construction schedules a value copy into a fresh destination timeline.
- Direct assignment detaches the destination onto fresh storage and a fresh queue.
- Previously enrolled buffers retain the old detached timeline.

### Async aliases

- Alias descriptors declare `async_alias_tag` or specialize `is_async_alias`.
- Structural copies preserve descriptor storage, lifetime owner, and exact queue.
- Mutable alias assignment requires ADL-visible `assign_through(target, source)`.
- Assignment writes through; it never retargets the descriptor.
- Read-only aliases reject assignment.
- Alias proxies do not expose descriptor mutation, `emplace`, storage access, or move-out.

## Async Tensor wrapper pattern

- Every caller-supplied Tensor operand in an async overload is an exact `Async<T>`.
- The non-coroutine wrapper creates buffer capabilities.
- The scheduled captureless coroutine owns those buffers and non-Tensor state by value.
- Immediate scalar parameters become ready value awaiters; async scalars contribute readers.
- Default backend selectors are determined from Tensor/storage types before scheduling.
- Runtime backend selection happens after Tensor values are awaited and mdspans exist.
- The coroutine calls the existing synchronous Tensor operation.
- An update output appears once as a writer; do not enroll it again as an input.
- Exact queue-identity rejection is only an obvious-alias check, not general overlap analysis.

## Exception and cancellation

- Unhandled coroutine failures propagate through output epochs.
- `WriteBuffer` coroutine parameters act as exception sinks.
- Cancellation is distinct from unconstructed storage and from an exception.
- Use `or_cancel()` where cancellation is an expected control path.

## Push-back triggers

- Captured coroutine state.
- A borrowed proxy escaping its buffer lifetime.
- Separate reader and writer used for ordinary mutation.
- Concurrent queue enrollment without external serialization.
- Scheduler order used as a correctness argument.
- Distinct overlapping objects treated as safe merely because both are `Async`.
