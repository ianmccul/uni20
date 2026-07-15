# Uni20 Async Runtime: AI Guidance

This file is for questions about `Async<T>`, `EpochQueue`, `ReadBuffer<T>`, `WriteBuffer<T>`, proxy lifetime, cancellation, and coroutine ordering.

## File-level invariants

- `Async<T>` uses `EpochQueue` for causality.
- `Async<T>` construction state comes from `shared_storage<T>`.
- `ReadBuffer<T>` and `WriteBuffer<T>` are the ordering primitives.
- `WriteBuffer<T>` is exclusive mutable access, not write-only access.
- Specialized awaiters are transport adaptors, not additional buffer capabilities.
- Scheduler timing does not define legality.
- Borrowed access and owning access are semantically different.
- Release ordering is semantically important.
- `Async<T>` does not solve aliasing across distinct objects.
- Async Tensor operation wrappers require every Tensor operand to be `Async<T>`;
  do not silently mix immediate and async operands.
- Scheduled Tensor wrappers move buffer handles and non-Tensor state into the
  coroutine, then call an existing synchronous Tensor operation after awaiting.

## Async<T>

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

### CAUSAL MODEL

- `read()` acquires `ReadBuffer<T>` for a specific epoch.
- `write()` acquires `WriteBuffer<T>` for the next epoch.
- For one queue, an ordinary operation uses any number of readers or one writer.
- Mutation reads the old value through the one writer; it does not acquire a separate reader.
- A separate reader and writer on one queue denote a deliberate two-epoch transition.

### LIFETIME / OWNERSHIP

- Buffers keep storage and their selected epoch context alive even if the originating `Async<T>` is moved or destroyed.

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
- `is_async_alias<T>`

## shared_storage<T>

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
- Alias storage owns a local descriptor and retains one reference to a separate lifetime owner.
- Copies of alias storage increment the alias shard, not the lifetime owner's count.
- The last alias destroys its local descriptor before releasing the owner reference.

### ALIAS ORDERING

- `make_async_alias(...)` shares the parent's exact `EpochQueue`.
- A separate queue for storage that aliases parent bytes is a correctness bug.
- Types declaring `async_alias_tag` make `is_async_alias_v<T>` true and receive
  structural `Async<T>` copy semantics.

### QUEUE ENROLLMENT THREADING CONTRACT

- Calls to `Async::read()` and `Async::write()` that share an `EpochQueue` must be externally serialized.
- This rule applies collectively to a parent and every alias using its queue.
- After a buffer is created, its task may execute concurrently; synchronization then belongs to `EpochContext`.
- Do not add queue-head locking merely to support concurrent enrollment unless the threading contract is deliberately redesigned.
- Whole-parent queue sharing may over-serialize disjoint aliases; solve that later with explicit subrange hazards, never independent queues over aliased bytes.

### FAILURE MODES

- Assuming valid storage implies a live `T` object.
- Moving a value out with `take()` and then continuing as if the stored object still exists.

### MISCONCEPTIONS

- Unconstructed `shared_storage<T>` behaves like a default-constructed `T`.
- Moving from stored state leaves a live but moved-from object behind.

### RELATED

- `Async<T>`
- unconstructed storage

## EpochQueue

### ROLE

- `EpochQueue` is the causal timeline for one `Async<T>`.
- `EpochContext` is one step in that timeline.

### INVARIANTS

- Conceptual order is `writer_n -> readers_n -> writer_{n+1} -> readers_{n+1} -> ...`.
- Scheduler run order does not replace `EpochQueue` order.

### CAUSAL MODEL

- `ReadBuffer<T>` refers to one `EpochContext` in the `EpochQueue`.
- `WriteBuffer<T>` refers to one `EpochContext` in the `EpochQueue`.
- A coroutine may await a valid, non-conflicting set of buffers in the order required by its data dependencies.
- `all(...)` may await several buffers together.
- The scheduler only runs tasks whose awaited buffers are ready.

### FAILURE MODES

- Acquiring both a reader and writer on one queue to model ordinary mutation.
- Expecting construction or await order to make a conflicting access set valid.
- Holding a reader while awaiting a later writer from the same queue in a deliberate two-epoch transition.

### MISCONCEPTIONS

- `EpochQueue` is just an implementation detail with no semantic role.
- Global task execution order defines legality.

### RELATED

- `Async<T>`
- `ReadBuffer<T>`
- `WriteBuffer<T>`

## Async Tensor operation wrappers

### ROLE

- Lift an existing synchronous Tensor operation onto the async runtime.
- Own synchronization and lifetime without making backend kernels async-aware.

### INVARIANTS

- Every caller-supplied Tensor operand is an exact `Async<T>`.
- The non-coroutine wrapper creates buffers; the coroutine owns those buffers by value.
- Immediate scalars become always-ready value awaiters; async scalars contribute
  real read buffers. Both are owned by the coroutine.
- Default selectors are resolved from Tensor/storage types before scheduling
  and enter the coroutine by value.
- The runtime backend walk occurs after the Tensor values are awaited and their
  mdspans are available.
- A one-output task carries its `WriteBuffer` as a coroutine parameter so
  unhandled failures propagate to that output epoch.
- Output/input queue identity may be rejected as an exact obvious-alias check;
  this is not general deadlock analysis.

### FAILURE MODES

- Capturing an `Async<T>` reference or caller-local configuration in a coroutine.
- Mixing synchronous Tensor operands into an async operation overload.
- Acquiring a read and write buffer for one output instead of mutating through
  its single writer.
- Inspecting an `Async<Tensor>` value during static selector resolution.
- Running layout- or accessor-dependent backend selection before awaited
  mdspans are available.
- Treating a by-value borrowed descriptor as owning storage.

### RELATED

- `docs/async/kernel_authoring.md`
- `src/uni20/linalg/async/`

## ReadBuffer<T>

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

## WriteBuffer<T>

### ROLE

- `WriteBuffer<T>` is the single exclusive-access capability for one epoch of `Async<T>`.

### INVARIANTS

- `WriteBuffer<T>` refers to exactly one write epoch.
- `WriteBuffer<T>` is move-only.
- `co_await writer` returns `WriteAccessProxy<T>`.
- `co_await writer.transfer()` returns `OwningWriteAccessProxy<T>`.
- Direct `co_await` on a temporary `WriteBuffer<T>` also uses the owning rvalue path.
- For independent values, the proxy may inspect, mutate, construct, replace, or
  move the value without a separate `ReadBuffer<T>`.
- For aliases, the proxy exposes the descriptor read-only and permits only
  assignment supported by ADL `assign_through`.
- Independent-value proxy assignment constructs empty storage and otherwise
  evaluates `stored_value = source`.
- Independent-value `operator+=` and `operator-=` may initialize unconstructed storage.

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
- Exposing a mutable alias descriptor, `emplace`, storage access, or move-out
  through an alias proxy and thereby permitting descriptor retargeting.
- Taking a separate reader and writer for the same queue when one writer should provide the mutation access.
- Failing to release the reader before the writer in an intentionally sequential two-epoch transition.

### MISCONCEPTIONS

- `WriteAccessProxy<T>` is just an ordinary `T&`.
- A mutable alias requires a different buffer type.
- `WriteBuffer<T>` grants write-only access and therefore needs a separate reader for mutation.
- First write always requires a separate `emplace(...)` call.

### RELATED

- `Async<T>`
- `EpochQueue`
- `ReadBuffer<T>`
- `is_async_alias<T>`
- `assign_through(...)`

## is_async_alias<T>

### ROLE

- `is_async_alias<T>` classifies an async payload as an independent value or a
  shared alias descriptor.

### INVARIANTS

- Ordinary types make `is_async_alias_v<T>` false.
- Types declaring `async_alias_tag`, or specializing the trait, make it true.
- Value copies create independent storage and a new queue, then schedule a value copy.
- Shared-alias copies retain descriptor storage, lifetime ownership, and the exact queue.

### FAILURE MODES

- Giving aliased bytes an independent queue.
- Reconstructing an alias descriptor so it no longer matches its retained owner and queue.

### MISCONCEPTIONS

- Alias classification controls the stored type's assignment expression.
- A shared alias is merely a copied raw pointer.

### RELATED

- `Async<T>`
- `async_alias_tag`
- `make_async_alias(...)`

## Async assignment capability

### ROLE

- Assignment behavior follows `is_async_alias_v<T>` and the operations
  available for the source type.

### INVARIANTS

- Ordinary values detach the destination handle and initialize fresh storage
  with a fresh `EpochQueue`.
- Mutable alias descriptors provide an ADL-visible
  `assign_through(target, source)` operation.
- Alias assignment retains descriptor storage, lifetime ownership, and the
  exact queue while invoking that operation.
- Exact `Async<T>` copy/move assignment follows the same capability rule.
- An alias without matching `assign_through` rejects that assignment source.
- Every `Async<T>` may expose `.write()` because it denotes exclusive epoch
  access; unsupported proxy operations are constrained away.
- `move_from_wait()` and synchronous mutable descriptor access remain
  unavailable for aliases.
- Copy/move construction of a shared alias remains valid and preserves the
  complete descriptor, owner, and queue identity.
- Alias proxies never expose `emplace`, storage access, arithmetic descriptor
  mutation, or move-out.

### MOTIVATING EXAMPLE

```cpp
Async<Tensor> x = make_first_tensor();
consume(x);
x = make_second_tensor(); // rebind: a new storage/queue branch
consume(x);

auto y = async::reshape_view(x, rows, columns); // owner-retaining Async<View>
Async<Tensor> values = make_replacement_values();
y = values;                                // heterogeneous Async<Tensor>: write through into x
y = async::reshape_view(z, rows, columns); // exact Async<View>: also write through into x

auto read_only = async::conj(x);
read_only = async::conj(z); // ill-formed: read-only aliases are not assignable
```

### FAILURE MODES

- Reconstructing only an alias descriptor so it no longer matches its retained
  owner and queue.
- Rebinding any shared alias, including for an exact-type assignment source.
- Implementing rebind by destroying and emplacing inside the old storage.
- Giving a write-through alias an independent queue.
- Implementing `assign_through` by retargeting the descriptor rather than
  mutating the referenced value.

### RELATED

- `Async<T>`
- `is_async_alias<T>`
- `assign_through(...)`
- `async_assign(...)`

## Write-proxy assignment

### ROLE

- For values, `co_await writer = rhs` initializes empty storage or assigns an existing `T`.
- For aliases, it invokes matching `assign_through` without replacing the descriptor.

### INVARIANTS

- The source must both construct `T` and make `stored_value = source` valid.
- Empty storage constructs `T` from the source.
- Constructed storage evaluates the underlying assignment expression.
- `emplace(...)` is the explicit reconstruction operation inside the current timeline.
- Alias proxy inspection returns `T const&`; replacement and move-out operations are absent.

### FAILURE MODES

- Expecting proxy assignment to reconstruct an existing assignable object.
- Using proxy assignment for a construct-only or assign-only source instead of
  the corresponding explicit operation.

### RELATED

- `WriteAccessProxy<T>`
- `emplace(...)`
- `assign_through(...)`

## or_cancel()

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
