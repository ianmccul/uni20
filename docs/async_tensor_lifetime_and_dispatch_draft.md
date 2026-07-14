# Async Tensor Lifetime And Dispatch Draft

**Status:** design notes for review. Dense async aliases, conjugating tensor
views, and the first all-async dense matrix-product wrappers are implemented;
`AsyncArray`, slice descriptors, and Python integration remain future work.

This note records the async-specific part of the tensor design discussion:
how `Async<Tensor>` should produce safe views/refs, how those views interact
with `AsyncArray`, and how async operands fit into kernel dispatch.

Related notes:

- [`async_storage.md`](async_storage.md) - `Async<T>` storage, value kinds, and
  write-proxy assignment.
- [`kernel_dispatch.md`](kernel_dispatch.md) - backend lists, operation tags,
  and the existing async/lowering split.
- [`tensor_dispatch_and_view_semantics_draft.md`](tensor_dispatch_and_view_semantics_draft.md)
  - synchronous tensor concepts, output semantics, and view/ref roles.
- [`block_tensor.md`](block_tensor.md) - block tensor structure and per-block
  storage policies.
- [`block_coalescing.md`](block_coalescing.md) - coalesced views over the same
  block backing allocation.

## Problem Statement

A resolved `stdex::mdspan` does not carry the ownership and epoch state required
by an async tensor value. Such a descriptor is sufficient only after a buffer
has been awaited and while the owning access proxy remains alive.

For example:

```cpp
Async<Tensor<double, 2>> A = Tensor<double, 2>{rows, cols};
auto S = async_slice(A, rows, cols);
schedule([](auto s_) static -> AsyncTask {
  auto s = co_await s_;
  kernel(s);
}(S.read()));
```

`S` cannot merely store a pointer into `A`. It must preserve:

- the lifetime of `A`'s storage control block
- the epoch or hazard state that orders reads and writes
- the slice descriptor needed to rebuild the resolved mdspan-like view after
  the await
- the storage domain/backend selector, including CUDA device placement when
  applicable

This is the async version of the synchronous split between `Tensor`,
`TensorRef`, and resolved mdspan-like views. The async handle has to be more
than a resolved view.

## Core Model

The async tensor model should keep four things separate:

```cpp
descriptor        // extents, strides, offset, dtype/domain metadata
owner token       // keeps the parent storage/backing allocation alive
epoch token       // read/write ordering and exception propagation
resolved view     // mdspan-like descriptor materialized after await
```

The resolved view is a short-lived object used by a leaf kernel. The async
tensor handle or alias is the durable object that can be stored, copied, and
passed to dispatch.

For `Async<Tensor>`, the owner token is the `Async` value's
`shared_storage<Tensor>` control block. For a slice/view of that tensor,
the alias must share that same owner token, or an equivalent subobject alias
token that keeps the parent control block alive.

For `AsyncArray`, the owner token is typically a shared backing allocation plus
a descriptor table. Each element handle adds an element/block epoch token.

## Async Aliases

Dense aliases use ordinary `Async<View>` types. The view models `TensorView`,
not mdspan, and therefore retains synchronous tensor metadata and exposes
`backend_selector()` plus `mdspan()`:

```cpp
Async<Tensor<complex<double>, 2>> parent;
Async<ConjugatedTensorView<Tensor<complex<double>, 2>>> view =
    async::conj(parent);
```

The alias `shared_storage<View>` owns the descriptor in a local control block.
That control block retains one reference to the parent storage, forming a
hierarchical reference-count shard. `Async<View>` also shares the parent's
exact `EpochQueue`. Alias descriptor types declare `async_alias_tag`, so copies
of `Async<View>` are structural handle copies rather than value copies onto a
new timeline. Alias-marked types cannot use ordinary root `Async<T>` value
constructors; they must be created with `make_async_alias(...)` or copied/moved
from an existing alias.

`read()` and `write()` return buffer-like awaitables. Awaiting them performs
the synchronization and then exposes the stored tensor-level descriptor. A
leaf operation obtains the resolved mdspan by calling `mdspan()` on that view:

```cpp
auto a = co_await A.read();       // resolved const view
auto s = co_await S.read();       // resolved const slice view
auto c = co_await C.write();      // resolved mutable output view
```

The resolved view should normally be treated as ephemeral. It is valid for the
scope in which the access proxy or owning await result keeps the epoch and owner
tokens alive. Persistent Python-facing or user-facing views need to own an
alias token, not just a pointer.

## Existing Async Facilities

The current `Async<T>` implementation already has the pieces this design should
build on:

- `shared_storage<T>` provides a refcounted control block with constructed and
  unconstructed states.
- `ReadBuffer<T>` and `WriteBuffer<T>` retain epoch context and storage
  ownership while reads/writes are pending.
- rvalue read/write awaits can return owning proxies, so an access epoch can
  outlive the original buffer object.
- `Async<T>::value_ptr()` creates a pointer-like owner token by capturing
  `shared_storage<T>`.
- `make_shared_storage_alias(...)` owns a local descriptor while retaining one
  parent-storage reference.
- `make_async_alias(...)` combines that ownership with the parent's exact queue.

That last point matters. A tensor slice/view alias must share the parent's
lifetime and must use the correct hazard/epoch context. A fresh independent
queue is not correct for a view that aliases the parent's bytes, because writes
through the parent and writes through the view must be ordered together.

The first dense implementation conservatively shares the whole parent queue.
Subrange hazard records remain future work for independently schedulable
slices and coalesced block views.

## AsyncArray

`AsyncArray` should be the same design at block granularity:

```cpp
one shared backing allocation
synchronous descriptor table
per-block epoch queues or hazard records
block(i) -> async tensor alias handle
```

`block(i)` should return a handle, not a raw resolved view. The handle keeps the
backing allocation alive, refers to the descriptor for block `i`, and uses the
epoch/hazard record for block `i`.

```cpp
auto bi = blocks.block(i);
auto v = co_await bi.read();   // materialize view after block i is ready
```

This lets `AsyncArray` avoid allocating a full `Async<T>` per block while still
presenting a common read/write interface to block kernels.

Coalesced views need a wider hazard record. A coalesced view spanning blocks
`i...j` cannot be protected by only one element queue. The first implementation
can choose a conservative policy:

- a block view awaits one block queue
- a coalesced view awaits every covered block queue, or a parent/coalesced
  queue that serializes against those blocks
- writes use the same hazard set as reads, but with writer exclusivity

This is the async side of the dual block-view/coalesced-view problem in
`block_coalescing.md`.

## Dispatch On Async Operands

Leaf kernels should not become async-aware. The async layer should lower async
operands to the same resolved mdspan-like views that synchronous dispatch uses.

The first implemented flow uses strict all-async Tensor operands:

1. Resolve the immutable selector from the Tensor/storage types and operation,
   or accept an explicit selector by value.
2. Reject exact output/input `EpochQueue` identity before buffer enrollment.
3. Normalize immediate or async scalar operands with `async::read(...)`.
4. Create one output `WriteBuffer` and input `ReadBuffer`s.
5. Move the selector, buffers, scalar awaiters, and ordinary configuration into
   a scheduled coroutine.
6. Await the stored Tensor values and async scalars inside the coroutine.
7. Call the existing synchronous Tensor operation with the selector. It
   performs shape preparation, mdspan resolution, and the runtime backend walk.

```cpp
void assign_product(Async<Matrix>& output,
                    Async<Matrix> const& lhs,
                    Async<Matrix> const& rhs,
                    auto&& alpha)
{
  auto selector = select_backend_for<Matrix, Matrix, Matrix>(gemm_op{});
  validate_obvious_queue_alias(output, lhs, rhs);
  schedule(assign_product_task(std::move(selector), output.write(), lhs.read(),
                               rhs.read(), async::read(alpha)));
}
```

The task is a named free coroutine, or equivalently a captureless `static`
lambda whose buffers are parameters. It never retains references to the
caller-owned `Async<T>` handles.

This is deliberately different from asking `Async<Tensor>` to satisfy the same
immediate `TensorView` concept as `Tensor`. A synchronous tensor can
produce an immediate read view. An async tensor produces a read handle that must
be awaited.

For whole `Async<Tensor>`, shape, layout, and data handles are part of the
stored Tensor value and become available after the await. The default selector
does not inspect those values: `select_backend_for<Tensors...>(operation)` uses
the common storage-policy type and operation value at submission. The runtime
backend walk still waits for resolved mdspans because layout and accessor checks
may decline a candidate. An explicit immutable selector is likewise passed into
the coroutine by value.

The public wrappers intentionally do not normalize mixed synchronous and async
Tensor operands. Every Tensor operand in one async operation is `Async<T>`.
Resolved `SpanLike` concepts remain leaf-kernel concepts.

## Shape Preparation And Outputs

`ensure_shape(out, shape)` remains the synchronous Tensor output hook, but a
whole `Async<Tensor>` output cannot use it until its writer epoch is active.

The implemented matrix-product distinction is:

- `assign_product` is an overwrite. It constructs an unconstructed output when
  the Tensor type is constructible from its extents, then delegates resizing or
  validation to the synchronous operation.
- `add_product` is an update. The output must already be constructed and have
  the correct shape because its old values participate in the result.

An async handle with synchronously known structure, such as a future
`AsyncArray` block descriptor, may validate shape at submission. Whole
`Async<Tensor>` remains structure-async: shape and layout metadata become
available only after await. Its storage-policy type is nevertheless sufficient
to resolve the default selector before scheduling.

## Async Temporaries

Temporary tensors still follow the synchronous backend-selector/storage-domain
rule from `tensor_dispatch_and_view_semantics_draft.md`: allocation is chosen
from the backend selector value plus a storage domain or factory, and filling
the temporary is a copy/evaluation kernel.

The async layer adds a timing choice:

```cpp
auto tmp = make_temporary_tensor(selector, domain, descriptor);       // immediate object
auto tmp = make_async_temporary_tensor(selector, domain, descriptor); // async handle
```

An immediate temporary is useful inside a coroutine after all inputs have been
awaited. It has ordinary lifetime and is destroyed when the coroutine frame no
longer needs it.

An async temporary is useful when later scheduled work must depend on the
temporary after the producing wrapper returns. In that case the temporary needs
the same ingredients as other async tensor aliases:

- owner token for the scratch allocation
- descriptor and storage domain
- epoch token for writes by the producer and reads by consumers
- backend selector value compatible with the storage domain, including
  structural runtime state such as CUDA device and workspace/allocator policy

Population of the temporary is not a special operation. It is a copy or
evaluation kernel scheduled through the same dispatch path:

```cpp
auto tmp = make_async_temporary_tensor(selector, domain, descriptor);
schedule_copy(tmp.write(), A.read());
schedule_compute(tmp.read(), C.write());
```

If only a resolved mdspan-like view and a stateless backend type are available,
an async temporary can be allocated only when the wrapper also has a storage
domain or temporary factory. A CUDA backend type by itself is not enough to
choose a device or allocator for scratch storage; the backend selector value or
adaptor must carry that state.

## Value Kinds and Assignment

Two independent mechanisms govern async values:

- `async_value_kind_of<T>` controls whether copying `Async<T>` creates an
  independent value timeline or structurally shares alias storage, lifetime
  ownership, and the epoch queue.
- The stored type's ordinary assignment expression controls write-proxy
  assignment when async storage already contains a `T`. Empty storage is
  constructed from the source instead.

There is no separate async assignment policy. The stored type defines what its
own assignment means:

```cpp
Async<Tensor>         // value kind: assignment follows Tensor's assignment expression
Async<TensorRef>      // assignment follows TensorRef's assignment expression
Async<mdspan-like>    // descriptor assignment; suitable only as a resolved temporary
Async<TensorAlias>    // shared_alias kind: handle copies retain owner/queue/descriptor
```

A persistent async tensor slice should probably be an alias handle, not a
stored resolved view. If a `TensorRef` type exists, an `Async<TensorRef>` value
must ensure that the `TensorRef` carries an owner token; otherwise it is just a
borrowed pointer with unsafe async lifetime.

This argues for distinguishing:

- **resolved views**: mdspan-like, descriptor assignment, short-lived
- **tensor refs**: write-through lvalue proxies, owner token required if stored
- **async aliases**: async handles whose `read()`/`write()` materialize resolved
  views after epoch synchronization

## CUDA Placement

Dense CUDA views and aliases must carry device placement explicitly. The owner
token keeps memory alive, but dispatch also needs to know:

- host versus device domain
- device ordinal
- stream/scheduler compatibility, when relevant
- whether peer access or staging is required

For synchronous tensors this can live in the storage/adaptor domain. For async
aliases it must be copied into or recoverable from the alias handle, because
the resolved mdspan-like view may contain only a pointer and layout.

## Python Boundary

Python-facing references have similar lifetime needs. A Python view into a
tensor cannot expose only a pointer and shape; it must hold an owner token that
keeps the parent tensor alive. For async tensors, the boundary likely should not
export a resolved view unless the value has been awaited. A persistent exported
view should own the same kind of alias token described above.

This matches Python reference-binding expectations more closely than raw
mdspan assignment semantics: copying the Python object copies the handle to the
same data, while explicit tensor assignment or NumPy-style writes define when
data is copied.

## Example Surfaces

### Async Dense Input

```cpp
Async<Tensor<double, 2>> A, B, C;

assign_product(C, A, B);
```

Expected behavior:

- all Tensor operands are async
- the wrapper moves `C.write()`, `A.read()`, and `B.read()` into a coroutine
- the original handles may disappear after submission because the buffers
  retain the selected storage and epochs
- the default selector is resolved from the Tensor/storage types before
  scheduling
- the runtime backend walk occurs after the stored Tensor values are awaited
- the synchronous Tensor operation resolves mdspans for leaf dispatch

### Async Slice

```cpp
auto S = async_slice(A, rows, cols);
assign_product(C, S, B);
```

Expected behavior:

- `S` keeps `A`'s storage alive
- `S` shares the correct parent hazard/epoch state
- `S.read()` awaits that hazard state and returns a resolved slice view
- using `S` as an output is a distinct mutable-alias design; read-only slice
  and conjugation aliases are input operands

### AsyncArray Block

```cpp
auto Ai = A.blocks().block(i);
auto Cj = C.blocks().block(j);
schedule_block_gemm(Cj, Ai, Bj);
```

Expected behavior:

- block descriptors are synchronous
- each block handle keeps shared backing storage alive
- each block handle uses the block epoch queue
- coalesced handles use a composite hazard set

## Tentative Conclusions

- Async tensor views need a durable alias handle that combines descriptor,
  owner token, epoch/hazard token, and backend/domain metadata.
- A raw resolved mdspan is not a safe async tensor value. It is a
  leaf-kernel argument materialized after await.
- `Async<Tensor>` should not be forced to satisfy the immediate synchronous
  `TensorView` concept. The first operation wrappers use exact `Async<T>`
  signatures rather than a broader async operand concept.
- `AsyncArray::block(i)` should return the same kind of async tensor alias
  handle, backed by shared allocation lifetime and per-block hazard state.
- `ensure_shape` remains the right output hook, but resizable
  `Async<Tensor>` outputs may need write access before shape preparation.
- Async Tensor wrappers require all Tensor operands to be async. Immediate and
  async scalar operands are normalized into awaiters; those awaiters, Tensor
  buffers, and ordinary state move into the coroutine.
- Static selector resolution happens before scheduling, while shape-sensitive
  preparation and the runtime backend walk happen after awaiting.
- Exact output/input queue identity is a useful early alias error, not a general
  deadlock detector or memory-overlap proof.
- Async temporaries need explicit backend-selector/storage-domain/factory
  information when that information cannot be recovered from the operand. The
  copy/evaluation into the temporary is an ordinary scheduled kernel.
- Fine-grained slice/coalesced hazards can start conservatively by serializing
  through the parent tensor or awaiting all covered block queues.

## Open Questions

1. **Generalized async operands.** After `AsyncArray` exists, is there a useful
   common async Tensor/block concept beyond the exact `Async<T>` overloads, or
   should wrappers remain explicit for each ownership and descriptor contract?
2. **Coalesced hazards.** Should a coalesced `AsyncArray` view await many block
   queues, or should `AsyncArray` have a parent/coalesced queue that composes
   with element queues?
3. **Synchronous block metadata.** Which future block handle metadata should be
   available before awaiting so contraction planning can occur at submission?
4. **Mutable alias surface.** How should a shared async alias expose mutation
   of referenced tensor elements without allowing its descriptor to be
   retargeted independently of the retained owner and epoch queue?
5. **Python view materialization.** How should a Python wrapper materialize an
   owning `Tensor` when an `Async<View>` is accessed, while keeping bare C++
   proxy objects out of the Python API?
6. **Async temporary type.** Should async temporaries be represented as ordinary
   `Async<Tensor<...>>`, as an `Async<TensorAlias>` over scratch storage, or
   as a separate scratch-buffer handle owned by the scheduler?
7. **Backend state.** Should async temporary allocation consume the same
    backend selector value used by dispatch, or only the composed backend state
    tuple fields that affect allocation, such as domain, device, scheduler
    target, allocator, communicator, and placement descriptor?
