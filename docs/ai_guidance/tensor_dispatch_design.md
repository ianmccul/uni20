# Tensor Dispatch and Backend-State Design: AI Guidance

This file is for AI assistants answering questions about the evolving tensor,
view, backend-dispatch, and temporary-allocation design.

## File-level answer rule

- Treat this file as roadmap / design draft guidance.
- Do not claim these APIs are implemented unless code has been inspected.
- Separate implemented tensor code from proposed dispatch concepts.
- Prefer "candidate design" and "tentative model" language.

## Authority

### RELATED DRAFTS

- `../tensor_dispatch_and_view_semantics_draft.md`
- `../async_tensor_lifetime_and_dispatch_draft.md`
- `../kernel_dispatch.md`
- `../backend_dispatch.md`

### STATUS

- Tensor/view/backend dispatch semantics are not finalized.
- The current design direction is concept/CPO-based dispatch, not inheritance
  from a central view type.
- `BasicTensor : TensorView` exists in current code, but it is not necessarily
  the long-term API contract.

## Tensor roles

### `BasicTensor`

- `ROLE`: Owning dense tensor value.
- `ASSIGNMENT`: Value/replace semantics.
- `OUTPUT`: May reuse storage or reallocate in `ensure_shape(...)`.
- `DO NOT CLAIM`: Do not claim `BasicTensor` assignment is mdspan-style rebind.

### `TensorRef`

- `ROLE`: Proposed non-owning write-through tensor lvalue, for slices or block
  outputs.
- `ASSIGNMENT`: Write-through into referenced storage when shape is compatible.
- `STATUS`: Proposed / design draft.

### resolved mdspan-like view

- `ROLE`: Leaf-kernel argument: data handle, extents, strides, accessor.
- `ASSIGNMENT`: mdspan-style descriptor rebind.
- `INVARIANT`: A resolved view is not enough for top-level Uni20 dispatch.

### `TensorView`

- `ROLE`: Current non-owning tensor/view abstraction.
- `STATUS`: Useful current type, but not necessarily the long-term central C++
  abstraction.
- `DO NOT CLAIM`: Do not claim every tensor should inherit from `TensorView`.

## Candidate tensor concepts

### DESIGN DIRECTION

Top-level tensor operations should use concepts/customization points instead of
base-class conversion to `TensorView`.

Candidate concepts:

- `TensorDescribed`: synchronous shape/layout/domain metadata.
- `TensorBackendSelectable`: default backend selector can be derived.
- `TensorReadable`: dispatchable tensor input.
- `TensorOutput`: output supports `ensure_shape(...)` and write access.
- `TensorReadWrite`: update operand that is both input and output.

### IMPORTANT DISTINCTION

- Leaf kernels use `SpanLike` / mdspan-like resolved views.
- Front-end Uni20 dispatch needs storage/backend/output metadata.
- A bare `stdex::mdspan` is suitable for a leaf kernel, but not enough for
  default top-level dispatch unless an explicit backend selector and state/domain
  are supplied.

## Backend dispatch

### OPERATION-TAG MODEL

- Backend dispatch is an ordered backend-list walk for an operation tag.
- `kernel_maybe_can(...)` is the type-level capability check.
- `try_kernel(...)` is the runtime attempt.
- A backend that lacks a usable `try_kernel(...)` overload is skipped by
  detection.

### STATELESS BACKEND TAGS

Current design direction: backend entries can be stateless tags.

```cpp
struct Device { int value; };
struct Stream { cudaStream_t value; };
struct CublasMathMode { math_mode_t value; };

struct CublasBackend {
  using state = std::tuple<Device, Stream, CublasMathMode>;
};

struct CudaGenericBackend {
  using state = std::tuple<Device, Stream>;
};
```

The selector state is the unique concatenation of backend state tuples:

```cpp
using state_t =
  unique_tuple_cat_t<CublasBackend::state, CudaGenericBackend::state>;
// std::tuple<Device, Stream, CublasMathMode>
```

### `unique_tuple_cat_t`

- `ROLE`: Uni20 helper, not standard C++.
- `MEANING`: Concatenate `std::tuple<...>` types and remove duplicate element
  types while preserving first occurrence order.
- `WHY`: `std::get<T>(tuple)` is valid only when `T` appears exactly once.
- `SCOPE`: Expected backend state tuples are short; simple implementation is
  acceptable unless compile-time profiling shows otherwise.

### STATE TAG RULES

- State tags are global/namespaced semantic names.
- Prefer specific tags: `cuda::Device`, `cuda::Stream`,
  `cublas::MathMode`, `cutensornetwork::WorkspaceLimit`.
- Do not reuse one tag type for semantically different state.
- Duplicate identical tags imply shared state.

## Temporaries

### DESIGN DIRECTION

Temporary allocation is separate from computation.

- Temporary type/storage comes from backend selector state plus storage domain
  or allocator/factory.
- Filling a temporary is an ordinary copy/evaluation kernel.
- Direct mdspan entry points that need temporaries must provide explicit backend
  selector state and storage-domain information.

### SAFE CLAIM

Do not say "backend type alone chooses temporary storage." CUDA temporaries need
state such as device and allocator.

## Async tensor aliases

### DESIGN DIRECTION

Async tensor views need durable alias handles, not just raw mdspan-like views.

An async tensor alias should preserve:

- owner token / storage lifetime
- epoch or hazard token
- descriptor / slice metadata
- backend state and storage-domain metadata

Resolved mdspan-like views should normally be materialized only after awaiting a
read/write handle.

## Python boundary

### DESIGN DIRECTION

Python-facing tensor views should keep an owner token. A Python view should not
expose only pointer plus shape if the parent tensor lifetime matters.

Uni20 may keep C++ tensor classes internal and use NumPy/nanobind/DLPack as the
boundary exchange type, but this is not finalized.

## Misconceptions

- `TensorView` being important today does not mean it must be the central
  long-term dispatch abstraction.
- Backend fallback order should not be encoded by inheritance.
- A CUDA backend tag by itself is not enough to allocate CUDA temporary storage.
- `std::tuple` can store duplicate types, but `std::get<T>` by type requires
  `T` to be unique.
- `std::type_list` does not exist in the C++ standard library.
- `unique_tuple_cat_t` is not a standard metafunction.
