# Uni20 Async Runtime: AI Guidance

This file is for questions about `Async<T>`, `EpochQueue`, `ReadBuffer<T>`, `WriteBuffer<T>`, proxy lifetime, cancellation, and coroutine ordering.

## File-level invariants

- `Async<T>` uses `EpochQueue` for causality.
- `Async<T>` construction state comes from `shared_storage<T>`.
- `ReadBuffer<T>` and `WriteBuffer<T>` are the ordering primitives.
- Scheduler timing does not define legality.
- Borrowed access and owning access are semantically different.
- Release ordering is semantically important.
- `Async<T>` does not solve aliasing across distinct objects.

## `Async<T>`

### ROLE

- `Async<T>` is the user-facing async value wrapper.
- `Async<T>` owns shared storage and an `EpochQueue` handle.

### INVARIANTS

- Each `Async<T>` has an ordered epoch timeline.
- `Async<T>()` means storage exists and the value is unconstructed.
- `Async<T>(args...)` means the value is constructed and immediately readable.
- `Async<T>` construction state is implemented by `shared_storage<T>`.
- First write may use `co_await writer = value`.
- First write may use `emplace(...)`.
- First write may use `rebind(...)`.

### CAUSAL MODEL

- `read()` acquires `ReadBuffer<T>` for a specific epoch.
- `write()` acquires `WriteBuffer<T>` for the next epoch.
- Buffer acquisition order matters.
- Await order matters.

### LIFETIME / OWNERSHIP

- Buffers keep storage and queue state alive even if the originating `Async<T>` is moved or destroyed.

### FAILURE MODES

- Reading or mutating through `Async<T>`-backed storage as if `Async<T>()` had already constructed a `T`.
- Using separate `Async<T>` objects over overlapping storage and assuming `Async<T>` makes that aliasing safe.

### MISCONCEPTIONS

- Scheduler timing determines async legality.
- `Async<T>` by itself proves memory-alias safety.

### RELATED

- `EpochQueue`
- `ReadBuffer<T>`
- `WriteBuffer<T>`
- `assignment_semantics_of<T>`

## `shared_storage<T>`

### ROLE

- `shared_storage<T>` is the internal reference-counted storage used by `Async<T>`.
- `shared_storage<T>` tracks control-block validity separately from value construction state.

### INVARIANTS

- `shared_storage<T>` may be valid while the contained `T` is unconstructed.
- `emplace(...)` always destroys any existing object before constructing a new `T`.
- `destroy()` destroys the current `T` but keeps the control block alive.
- `take()` moves the current `T` out and then destroys the stored object.
- After `take()`, the control block may remain valid while the value is again unconstructed.

### LIFETIME / OWNERSHIP

- `shared_storage<T>` uses reference-counted shared ownership of the control block.
- Construction state is a property of the contained object, not of handle validity.

### FAILURE MODES

- Assuming valid storage implies a live `T` object.
- Moving a value out with `take()` and then continuing as if the stored object still exists.

### MISCONCEPTIONS

- Unconstructed `shared_storage<T>` behaves like a default-constructed `T`.
- Moving from stored state leaves a live but moved-from object behind.

### RELATED

- `Async<T>`
- unconstructed storage

## `EpochQueue`

### ROLE

- `EpochQueue` is the causal timeline for one `Async<T>`.
- `EpochContext` is one step in that timeline.

### INVARIANTS

- Conceptual order is `writer_n -> readers_n -> writer_{n+1} -> readers_{n+1} -> ...`.
- Scheduler run order does not replace `EpochQueue` order.

### CAUSAL MODEL

- `ReadBuffer<T>` refers to one `EpochContext` in the `EpochQueue`.
- `WriteBuffer<T>` refers to one `EpochContext` in the `EpochQueue`.
- A coroutine may await buffers in any order.
- `all(...)` may await several buffers together.
- The scheduler only runs tasks whose awaited buffers are ready.

### FAILURE MODES

- Acquiring or awaiting buffers in the wrong order and then expecting scheduler timing to recover correctness.
- Holding a read gate longer than necessary and then awaiting a conflicting write.

### MISCONCEPTIONS

- `EpochQueue` is just an implementation detail with no semantic role.
- Global task execution order defines legality.

### RELATED

- `Async<T>`
- `ReadBuffer<T>`
- `WriteBuffer<T>`

## `ReadBuffer<T>`

### ROLE

- `ReadBuffer<T>` is the capability object for reading one epoch of `Async<T>`.

### INVARIANTS

- `ReadBuffer<T>` refers to exactly one epoch.
- `co_await reader` returns `T const&`.
- `co_await reader.transfer()` returns `OwningReadAccessProxy<T>`.
- Direct `co_await` on a temporary `ReadBuffer<T>` also uses the owning rvalue path.
- `reader.maybe()` is the optional-read path.
- `reader.or_cancel()` is the cancellation-aware read path.

### CAUSAL MODEL

- `ReadBuffer<T>` establishes dependence on the writer of that epoch.
- `ReadBuffer<T>` participates in `writer_n -> readers_n` ordering.

### LIFETIME / OWNERSHIP

- Borrowed read lifetime is tied to the `ReadBuffer<T>` object.
- Owning read lifetime is tied to the `OwningReadAccessProxy<T>` object.
- `release()` explicitly drops the read epoch.
- `get_release()` reads and drops the read epoch in one step.

### FAILURE MODES

- Using a borrowed read reference after the `ReadBuffer<T>` lifetime ends.
- Keeping a read alive across a conflicting write when the read value is no longer needed.

### MISCONCEPTIONS

- The reference from `co_await reader` is independent of the `ReadBuffer<T>` lifetime.
- Releasing a read is only a performance detail.

### RELATED

- `Async<T>`
- `EpochQueue`
- `WriteBuffer<T>`
- `or_cancel()`

## `WriteBuffer<T>`

### ROLE

- `WriteBuffer<T>` is the capability object for writing one epoch of `Async<T>`.

### INVARIANTS

- `WriteBuffer<T>` refers to exactly one write epoch.
- `WriteBuffer<T>` is move-only.
- `co_await writer` returns `WriteAccessProxy<T>`.
- `co_await writer.transfer()` returns `OwningWriteAccessProxy<T>`.
- Direct `co_await` on a temporary `WriteBuffer<T>` also uses the owning rvalue path.
- Proxy assignment may construct first-write storage for `rebind` types.
- `operator+=` and `operator-=` may initialize unconstructed storage.

### CAUSAL MODEL

- Awaiting `WriteBuffer<T>` gates mutation until that write epoch becomes active.
- Completing the write opens the reader phase for that epoch.
- Completing the reader phase allows the next writer epoch.

### LIFETIME / OWNERSHIP

- Borrowed write proxy lifetime is tied to the `WriteBuffer<T>` object.
- Owning write proxy lifetime is tied to the `OwningWriteAccessProxy<T>` object.
- `return co_await writer.transfer()` is the explicit owning-transfer form for a named buffer.
- `return co_await writer` is not the safe form to recommend.

### FAILURE MODES

- Using a borrowed write proxy after the `WriteBuffer<T>` lifetime ends.
- Requesting mutable-reference-style access before construction and hitting `buffer_write_uninitialized`.
- Awaiting a write before releasing a conflicting read in a self-dependent kernel.

### MISCONCEPTIONS

- `WriteAccessProxy<T>` is just an ordinary `T&`.
- First write always requires a separate `emplace(...)` call.

### RELATED

- `Async<T>`
- `EpochQueue`
- `ReadBuffer<T>`
- `assignment_semantics_of<T>`

## `assignment_semantics_of<T>`

### ROLE

- `assignment_semantics_of<T>` controls what `co_await writer = rhs` means for `T`.

### INVARIANTS

- Default semantics are `rebind`.
- Opt-in semantics are `write_through`.
- `rebind` means proxy assignment reconstructs or replaces the stored object.
- `write_through` means proxy assignment writes through the existing object.

### FAILURE MODES

- Specializing `assignment_semantics_of<T>` for the wrong write behavior.
- Using proxy assignment expecting reconstruction on a `write_through` type or write-through on a `rebind` type.

### MISCONCEPTIONS

- `co_await writer = rhs` has one universal meaning for every `T`.

### RELATED

- `WriteBuffer<T>`
- `rebind`
- `write_through`

## `or_cancel()`

### ROLE

- `or_cancel()` is the explicit read form for cancellation-aware code paths.

### INVARIANTS

- `or_cancel()` surfaces cancellation as `task_cancelled`.
- `or_cancel()` is the correct read form when cancellation is an expected control path.

### FAILURE MODES

- Using plain read access where cancellation should terminate the branch explicitly.

### MISCONCEPTIONS

- Cancellation is the same as uninitialized storage.

### RELATED

- `ReadBuffer<T>`
- `task_cancelled`

## Exception propagation

### INVARIANTS

- Exceptions propagate through the async graph.
- `WriteBuffer<T>` coroutine parameters act as exception sinks.
- Downstream readers and writers observe propagated failures when awaited.

## Aliasing limit

### INVARIANTS

- `Async<T>` sequences epochs of one async object.
- `Async<T>` does not prove alias safety across distinct objects.
- `TensorView` overlap is a higher-level problem.
- `Async<TensorView>` or similar wrappers do not automatically make aliasing safe.

## Code-generation caution

### INVARIANTS

- GCC 13 rejects some dense nested `co_await` expressions that newer compilers accept.
- For portable Uni20 code suggestions, prefer a named temporary when the one-line form is parser-sensitive.

## Push-back triggers

- Push back if a proposal relies on scheduler timing rather than `EpochQueue` ordering.
- Push back if a proposal captures coroutine state instead of passing parameters.
- Push back if a proposal returns a borrowed proxy beyond the buffer lifetime.
- Push back if a proposal assumes alias safety that `Async<T>` does not provide.
- Push back if a proposal infers causality from execution order instead of buffer ordering.

## Related detailed docs

- `../async/runtime_model.md`
- `../async/buffers_and_awaiters.md`
- `../async/exceptions_and_cancellation.md`
- `../async/quick_reference.md`
