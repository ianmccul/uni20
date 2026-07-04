# Uni20 Glossary for AI Assistants

This glossary is optimized for retrieval, not pedagogy.

## Core async terms

### Async<T>

- `ROLE`: User-facing async value wrapper.
- `INVARIANT`: `Async<T>()` is unconstructed; `Async<T>(args...)` is constructed.
- `FAILURE MODE`: Using `Async<T>()` as if a `T` already exists.
- `RELATED`: `EpochQueue`, `ReadBuffer<T>`, `WriteBuffer<T>`.

### shared_storage<T>

- `ROLE`: Internal reference-counted storage used by `Async<T>`.
- `INVARIANT`: Control-block validity and object construction state are separate.
- `INVARIANT`: `emplace(...)` replaces by destroy-then-construct; `take()` moves out and destroys.
- `MISCONCEPTION`: Valid storage implies a live `T`.

### EpochQueue

- `ROLE`: Causal timeline for one `Async<T>`.
- `INVARIANT`: Conceptual order is `writer_n -> readers_n -> writer_{n+1} -> ...`.
- `FAILURE MODE`: Acquiring or awaiting buffers in the wrong order and expecting scheduler timing to recover correctness.
- `MISCONCEPTION`: Scheduler timing defines legality.

### EpochContext

- `ROLE`: One step in an `EpochQueue`.
- `RELATED`: `EpochQueue`, `ReadBuffer<T>`, `WriteBuffer<T>`.

### ReadBuffer<T>

- `ROLE`: Capability object for reading one epoch of an `Async<T>`.
- `CAUSAL MODEL`: Establishes dependence on the writer of that epoch.
- `LIFETIME / OWNERSHIP`: Borrowed read is tied to the `ReadBuffer<T>` object. Owning read is tied to `OwningReadAccessProxy<T>`.
- `FAILURE MODE`: Keeping a read alive across a conflicting write when the read is no longer needed.
- `MISCONCEPTION`: A borrowed read reference is independent of the `ReadBuffer<T>` lifetime.

### WriteBuffer<T>

- `ROLE`: Capability object for writing one epoch of an `Async<T>`.
- `CAUSAL MODEL`: Gates mutation until the write epoch is active.
- `LIFETIME / OWNERSHIP`: Borrowed write is tied to the `WriteBuffer<T>` object. Owning write is tied to `OwningWriteAccessProxy<T>`.
- `FAILURE MODE`: Awaiting a write before releasing a conflicting read in a self-dependent kernel.
- `MISCONCEPTION`: `WriteAccessProxy<T>` is just an ordinary `T&`.

### borrowed read

- `ROLE`: `co_await reader` on an lvalue `ReadBuffer<T>`.
- `INVARIANT`: Returns `T const&`.
- `LIFETIME / OWNERSHIP`: Lifetime is tied to the `ReadBuffer<T>` object.

### owning read

- `ROLE`: `co_await reader.transfer()`.
- `INVARIANT`: Returns `OwningReadAccessProxy<T>`.
- `LIFETIME / OWNERSHIP`: `OwningReadAccessProxy<T>` keeps the read epoch alive until release or destruction.

### borrowed write

- `ROLE`: `co_await writer` on an lvalue `WriteBuffer<T>`.
- `INVARIANT`: Returns `WriteAccessProxy<T>`.
- `LIFETIME / OWNERSHIP`: Lifetime is tied to the `WriteBuffer<T>` object.

### owning write

- `ROLE`: `co_await writer.transfer()`.
- `INVARIANT`: Returns `OwningWriteAccessProxy<T>`.
- `LIFETIME / OWNERSHIP`: `OwningWriteAccessProxy<T>` owns the writer handle.

### all(...)

- `ROLE`: Await helper that waits for several awaitables and returns a tuple of results.
- `RELATED`: `EpochQueue`, `ReadBuffer<T>`, `WriteBuffer<T>`.

### try_await(...)

- `ROLE`: Readiness probe for an awaitable without fully blocking on it.

## Storage and assignment terms

### unconstructed storage

- `ROLE`: State of `Async<T>()` before first construction.
- `INVARIANT`: Storage exists but the contained `T` value does not yet exist.

### assignment_semantics_of<T>

- `ROLE`: Trait that controls what `WriteBuffer<T>` proxy assignment means for a type.
- `INVARIANT`: Default is `rebind`; opt-in alternative is `write_through`.
- `MISCONCEPTION`: `co_await writer = rhs` has one universal meaning for every `T`.

### rebind

- `ROLE`: Default proxy-assignment semantics.
- `INVARIANT`: `co_await writer = rhs` reconstructs or replaces the stored object.

### write_through

- `ROLE`: Opt-in proxy-assignment semantics for reference-like or proxy-like types.
- `INVARIANT`: Assignment writes through the existing object instead of reseating it.

### rebind(...)

- `ROLE`: Explicit write-proxy operation that reconstructs or retargets the stored object.

### take()

- `ROLE`: Move the stored value out of a write target and destroy the stored object.

### take_release()

- `ROLE`: `take()` plus explicit writer release.

### get_release()

- `ROLE`: Owning read-proxy operation that returns the value and releases the read epoch in one step.
- `RELATED`: `OwningReadAccessProxy<T>`, `ReadBuffer<T>`.

## AD terms

### Var<T>

- `ROLE`: User-facing reverse-mode variable.
- `INVARIANT`: Combines a forward `Async<T>` value and a reverse `ReverseValue<T>` accumulation channel.
- `LIFETIME / OWNERSHIP`: Owns its forward channel and reverse channel by value.
- `MISCONCEPTION`: `Var<T>` is just the old `Dual<T>` name or a tape node.

### ReverseValue<T>

- `ROLE`: Gradient accumulation helper used by `Var<T>`.
- `INVARIANT`: `ReverseValue<T>` exposes async reads and writes for reverse-mode propagation.
- `LIFETIME / OWNERSHIP`: Owns the internal async gradient channel.
- `FAILURE MODE`: Treating gradient accumulation as if it were unordered by default.
- `MISCONCEPTION`: `ReverseValue<T>` is just a passive numeric field with no async semantics.

### backprop()

- `ROLE`: Exposes the async gradient channel for a variable or reverse value.
- `LIFETIME / OWNERSHIP`: Lvalue overloads return references to the owned finalized channel; rvalue overload moves it out.
- `FAILURE MODE`: Waiting on `backprop()` with no seeded downstream gradient and assuming a value must already exist.
- `MISCONCEPTION`: `backprop()` launches a separate global backward phase or replays a tape.

### gradient materialization

- `ROLE`: Point at which a gradient becomes concrete.
- `INVARIANT`: Requires upstream seeding and reverse propagation.
- `MISCONCEPTION`: Every gradient exists eagerly from the start.

### Wirtinger dL/dz*

- `ROLE`: Complex-gradient convention used by Uni20 for real scalar losses.
- `INVARIANT`: Uni20 uses `dL/dz*`.

## Scheduler terms

### DebugScheduler

- `ROLE`: Deterministic scheduler used mainly for tests and runtime debugging.

### TbbScheduler

- `ROLE`: General parallel scheduler built on oneTBB.

### TbbNumaScheduler

- `ROLE`: NUMA-aware scheduler built on oneTBB.

### ScopedScheduler

- `ROLE`: Temporary scheduler override.
- `INVARIANT`: Mostly useful in tests and controlled experiments, not normal user code.

## Tensor and aliasing terms

### BasicTensor

- `ROLE`: Owning dense tensor value.
- `STATUS`: Current type, but long-term relation to `TensorView` is still design work.
- `ASSIGNMENT`: Should be reasoned about as value/replace semantics, not mdspan-style descriptor rebind.

### TensorView

- `ROLE`: Non-owning tensor/view abstraction.
- `INVARIANT`: `TensorView` is important, but its ownership and async interaction rules are still evolving.
- `MISCONCEPTION`: `TensorView` being important means every top-level tensor dispatch API should name `TensorView`.

### TensorRef

- `ROLE`: Proposed non-owning write-through tensor lvalue for slices or block outputs.
- `STATUS`: Design draft, not settled API.
- `ASSIGNMENT`: Writes through to referenced storage when shape is compatible.

### resolved mdspan-like view

- `ROLE`: Leaf-kernel argument containing data handle, extents, strides, and accessor.
- `INVARIANT`: Suitable for leaf kernels after backend compatibility is known.
- `MISCONCEPTION`: A resolved mdspan-like view carries enough metadata for default Uni20 top-level dispatch.

### aliasing

- `ROLE`: Two handles refer to overlapping storage.
- `INVARIANT`: Uni20 async ordering does not automatically solve aliasing correctness across distinct async objects or views.
- `MISCONCEPTION`: Wrapping a view-like object in `Async<T>` automatically solves overlap ordering.

## Backend dispatch terms

### backend tag

- `ROLE`: Stateless type representing a candidate backend in an ordered backend list.
- `INVARIANT`: Backend order comes from `backend_list<...>`, not inheritance.
- `RELATED`: backend state tag, backend selector.

### backend state tag

- `ROLE`: Runtime state component required by one or more backend tags.
- `EXAMPLES`: `cuda::Device`, `cuda::Stream`, `cublas::MathMode`.
- `INVARIANT`: State tag types should have global/namespaced semantic meaning.
- `MISCONCEPTION`: One generic tag name can safely mean different things for different backends.

### backend selector

- `ROLE`: Value that combines ordered backend tags with composed runtime backend state.
- `STATUS`: Design draft.
- `INVARIANT`: Backend selector state is composed from backend tag `using state = std::tuple<...>` declarations.

### unique_tuple_cat_t

- `ROLE`: Proposed Uni20 helper for composing backend state tuples.
- `STATUS`: Not a standard C++ metafunction.
- `MEANING`: Concatenate `std::tuple<...>` types and remove duplicate element types.
- `WHY`: `std::get<T>(tuple)` by type requires `T` to appear exactly once.

### std::type_list

- `STATUS`: Does not exist in the C++ standard library.
- `GUIDANCE`: Use `std::tuple<...>` or a Uni20-local helper when a type pack must be named.

## Safety terms

### coroutine safety rule

- `ROLE`: Hard safety rule for coroutine lambdas.
- `INVARIANT`: Coroutine lambdas that return Uni20 async task types must be captureless and `static`.
- `INVARIANT`: Values must be passed as parameters, not captured.

### exception sink

- `ROLE`: Buffer or epoch that receives propagated exceptions from an upstream coroutine.

### cancellation

- `ROLE`: Terminal state meaning a value will not become available normally.
- `INVARIANT`: `or_cancel()` is the explicit read form that surfaces cancellation as `task_cancelled`.
