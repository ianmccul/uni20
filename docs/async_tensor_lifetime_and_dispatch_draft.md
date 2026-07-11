# Async Tensor Lifetime And Dispatch Draft

**Status:** design notes for review. This is not a finalized API contract.

This note records the async-specific part of the tensor design discussion:
how `Async<Tensor>` should produce safe views/refs, how those views interact
with `AsyncArray`, and how async operands fit into kernel dispatch.

Related notes:

- [`async_storage.md`](async_storage.md) - `Async<T>` storage and write-proxy
  assignment semantics.
- [`kernel_dispatch.md`](kernel_dispatch.md) - backend lists, operation tags,
  and the existing async/lowering split.
- [`tensor_dispatch_and_view_semantics_draft.md`](tensor_dispatch_and_view_semantics_draft.md)
  - synchronous tensor concepts, output semantics, and view/ref roles.
- [`block_tensor.md`](block_tensor.md) - block tensor structure and per-block
  storage policies.
- [`block_coalescing.md`](block_coalescing.md) - coalesced views over the same
  block backing allocation.

## Problem Statement

A raw `BasicTensorView` adaptor or resolved `stdex::mdspan` does not carry the
ownership and epoch state required by an async tensor value. Such a descriptor
is sufficient only after a buffer has been awaited and while the owning access
proxy remains alive.

For example:

```cpp
Async<BasicTensor<double, 2>> A = make_tensor();
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

This is the async version of the synchronous split between `BasicTensor`,
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

For `Async<BasicTensor>`, the owner token is the `Async` value's
`shared_storage<BasicTensor>` control block. For a slice/view of that tensor,
the alias must share that same owner token, or an equivalent subobject alias
token that keeps the parent control block alive.

For `AsyncArray`, the owner token is typically a shared backing allocation plus
a descriptor table. Each element handle adds an element/block epoch token.

## Async Aliases

The design likely needs an explicit async alias handle. Possible names:

```cpp
AsyncTensorRef
AsyncTensorAlias
TensorAccess
AsyncTensorView
```

The name is open, but the role is not. An async alias is not just
`Async<mdspan>` and not just `Async<BasicTensorView>`. It is a small handle that
contains enough information to materialize a view safely later:

```cpp
template <class Parent, class Descriptor, class Hazard>
class AsyncTensorAlias {
  parent_owner_token owner_;
  Descriptor descriptor_;
  Hazard hazard_;
  backend_selector_type selector_;

public:
  auto read() const;  // awaitable -> resolved const mdspan-like view
  auto write();       // awaitable -> resolved mutable mdspan-like view/ref
};
```

`read()` and `write()` do not return the final view immediately. They return
buffer-like awaitables. Awaiting those objects performs the synchronization and
then combines the current parent data pointer with the stored descriptor:

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
- The existing `deferred` constructors sketch aliasing, but they currently do
  not fully specify tensor subobject aliasing or queue sharing.

That last point matters. A tensor slice/view alias must share the parent's
lifetime and must use the correct hazard/epoch context. A fresh independent
queue is not correct for a view that aliases the parent's bytes, because writes
through the parent and writes through the view must be ordered together.

The current `shared_storage` documentation also notes that there is not yet a
facility for sharing ownership with subobjects. Async tensor aliases are the
main use case for such a facility.

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

The dispatch flow is:

1. Read synchronous descriptors and backend selectors from the tensor operands.
2. Select a backend from the operation tag and backend list.
3. Schedule a coroutine when at least one operand is async.
4. Inside the coroutine, await read/write handles and materialize resolved views.
5. Call the already-selected backend or leaf kernel on those resolved views.

Schematic:

```cpp
template <class C, class A, class B>
auto gemm(C& c, A const& a, B const& b)
{
  auto shape = gemm_shape(tensor_descriptor(a), tensor_descriptor(b));
  auto selector = common_backend_selector(c, a, b);

  return dispatch_async_or_sync(selector, gemm_op{}, c, a, b, shape);
}
```

For an async path:

```cpp
schedule([](auto c_, auto a_, auto b_) static -> AsyncTask {
  auto C = co_await c_.write();
  auto A = co_await a_.read();
  auto B = co_await b_.read();
  selected_backend_gemm(C, A, B);
}(c, a, b));
```

This is deliberately different from asking `Async<Tensor>` to satisfy the same
immediate `TensorView` concept as `BasicTensor`. A synchronous tensor can
produce an immediate read view. An async tensor produces a read handle that must
be awaited.

The common concept should therefore be at the operand/access level, not at the
resolved-view level. Candidate split:

```cpp
TensorDescribed          // synchronous descriptor metadata
TensorBackendSelectable  // synchronous default backend selector
TensorReadAccess         // read() returns immediate view or awaitable handle
TensorWriteAccess        // write() returns immediate view/ref or awaitable handle
```

Then a helper can normalize immediate and awaitable access in the wrapper layer.
The exact names are open; the important point is that resolved `SpanLike`
concepts remain leaf-kernel concepts.

## Shape Preparation And Outputs

`ensure_shape(out, shape)` still makes sense for async outputs, but the timing
depends on whether the output structure is synchronous.

For an async handle whose structure is known synchronously, `ensure_shape` can
run at submission time if it only validates descriptors. This is the natural
case for:

- `AsyncArray` block handles
- fixed async tensor aliases
- block/coalesced views with known descriptors

For `Async<BasicTensor>`, resizing may require write access to the stored tensor:

```cpp
schedule([](auto c_, shape_type shape) static -> AsyncTask {
  auto C_owner = co_await c_.write();
  ensure_shape(C_owner.get(), shape); // may reallocate BasicTensor storage
  auto C = tensor_write_view(C_owner.get());
  kernel(C);
}(C.write(), shape));
```

That means a value-producing operation on an unconstructed or resizable
`Async<BasicTensor>` may not be able to materialize the output view until after
the write await. If backend selection needs output shape/domain metadata before
awaiting, the async tensor object must carry a synchronous descriptor or the
operation must be treated as structure-async.

This gives two async output categories:

```cpp
sync-structure async data   // descriptors known before scheduling
structure-async tensor      // descriptor becomes known only after await
```

`AsyncArray` and most block-tensor per-block data should live in the first
category. `Async<BasicTensor>` as the result of a truncation or shape-changing
operation may live in the second.

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

## Assignment Semantics

The async assignment trait in `async_storage.md` should remain about write
proxy assignment into `Async<T>` storage. Tensor types then choose the meaning
of `T::operator=`:

```cpp
Async<BasicTensor>     // value/replace: construct if empty, otherwise assign tensor value
Async<TensorRef>       // write-through: target must already refer to parent storage
Async<mdspan-like>     // rebind descriptor: copy/emplace descriptor, no element copy
AsyncTensorAlias       // handle semantics: copy aliases owner/hazard/descriptor
```

A persistent async tensor slice should probably be an alias handle, not a
stored resolved view. If a `TensorRef` type exists, an `Async<TensorRef>` value
must ensure that the `TensorRef` carries an owner token; otherwise it is just a
borrowed pointer with unsafe async lifetime.

This argues for distinguishing:

- **resolved views**: mdspan-like, rebinding assignment, short-lived
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
Async<BasicTensor<double, 2>> A;
BasicTensor<double, 2> B, C;

gemm(C, A, B);
```

Expected behavior:

- descriptors/backend selectors are read synchronously if available
- backend is selected before scheduling
- scheduled wrapper awaits `A.read()` and resolves a view
- `B` and `C` use immediate or normalized access handles
- leaf kernel sees resolved mdspan-like views only

### Async Slice

```cpp
auto S = async_slice(A, rows, cols);
gemm(C, S, B);
```

Expected behavior:

- `S` keeps `A`'s storage alive
- `S` shares the correct parent hazard/epoch state
- `S.read()` awaits that hazard state and returns a resolved slice view
- assigning to `S` writes through only if the shape is compatible

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
  `TensorView` concept. It should satisfy an async operand/access
  concept whose reads and writes are awaitable.
- `AsyncArray::block(i)` should return the same kind of async tensor alias
  handle, backed by shared allocation lifetime and per-block hazard state.
- `ensure_shape` remains the right output hook, but resizable
  `Async<BasicTensor>` outputs may need write access before shape preparation.
- Async temporaries need explicit backend-selector/storage-domain/factory
  information when that information cannot be recovered from the operand. The
  copy/evaluation into the temporary is an ordinary scheduled kernel.
- Fine-grained slice/coalesced hazards can start conservatively by serializing
  through the parent tensor or awaiting all covered block queues.

## Open Questions

1. **Names.** Should the durable async view handle be called `AsyncTensorRef`,
   `AsyncTensorAlias`, `TensorAccess`, or something else?
2. **Concept split.** Should the public concepts be `TensorReadAccess` /
   `TensorWriteAccess`, `AsyncTensorReadable` / `AsyncTensorWritable`, or a
   single operand concept with normalized `read()`/`write()` access?
3. **Subobject owner token.** Should `shared_storage` gain an explicit aliasing
   facility for subobjects, or should tensor aliases store a parent
   `shared_storage<Parent>` plus an offset/descriptor?
4. **Queue sharing.** For dense slices, should the first implementation share
   the parent tensor's whole epoch queue, or introduce slice/subrange hazard
   records immediately?
5. **Coalesced hazards.** Should a coalesced `AsyncArray` view await many block
   queues, or should `AsyncArray` have a parent/coalesced queue that composes
   with element queues?
6. **Structure-async dispatch.** How should backend selection work when an
   `Async<BasicTensor>` result does not have a descriptor until after awaiting
   the producer?
7. **Assignment trait.** Is the existing `rebind` versus `write_through` split
   sufficient once `BasicTensor::operator=` has value/replace semantics, or is
   a separate `value` assignment-semantic name still useful for documentation?
8. **Python view lifetime.** Should Python-facing views reuse the same alias
   handle type as C++ async tensor refs, or should they have a separate
   nanobind/DLPack owner-token wrapper?
9. **Async temporary type.** Should async temporaries be represented as ordinary
   `Async<BasicTensor<...>>`, as an `AsyncTensorAlias` over scratch storage, or
   as a separate scratch-buffer handle owned by the scheduler?
10. **Backend state.** Should async temporary allocation consume the same
    backend selector value used by dispatch, or only the composed backend state
    tuple fields that affect allocation, such as domain, device, scheduler
    target, allocator, communicator, and placement descriptor?
