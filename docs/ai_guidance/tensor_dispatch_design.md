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
- `../mdspan_linalg_dispatch_plan.md`

### STATUS

- Tensor/view/backend dispatch semantics are not finalized.
- The current design direction is concept/CPO-based dispatch, not inheritance
  from a central view type.
- `BasicTensor : TensorView` exists in current code, but it is not necessarily
  the long-term API contract.

## Tensor roles

### BasicTensor

- `ROLE`: Owning dense tensor value.
- `ASSIGNMENT`: Value/replace semantics.
- `OUTPUT`: May reuse storage or reallocate in `ensure_shape(...)`.
- `DO NOT CLAIM`: Do not claim `BasicTensor` assignment is mdspan-style rebind.

### TensorRef

- `ROLE`: Proposed non-owning write-through tensor lvalue, for slices or block
  outputs.
- `ASSIGNMENT`: Write-through into referenced storage when shape is compatible.
- `STATUS`: Proposed / design draft.

### resolved mdspan-like view

- `ROLE`: Leaf-kernel argument: data handle, extents, strides, accessor.
- `ASSIGNMENT`: mdspan-style descriptor rebind.
- `INVARIANT`: A resolved view is not enough for top-level Uni20 dispatch.

### TensorView

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
- `data_handle_type` being a pointer does not prove direct readability or
  writeability. Mdspan accessors define value semantics. Direct BLAS/LAPACK
  paths may bypass `access(...)` only for `stdex::default_accessor` or for
  accessors whose semantics are explicitly recognized and lowered, such as
  Uni20's C++26-style `conjugated_accessor` into BLAS transform metadata.
- Do not claim arbitrary custom accessors, transform accessors, zip accessors,
  or scaling accessors are BLAS-addressable merely because their data handle is
  pointer-like. They require materialization, generic evaluation, or an
  operation-specific lowering rule.
- Uni20 uses `uni20::conj(span)` for lazy conjugating mdspan views. This follows
  the C++26 `std::linalg::conjugated_accessor` model, but Uni20 does not adopt
  `conj-if-needed`; `uni20::conj` is the project-level fix for real-scalar
  conjugation semantics.

### PARAMETER ORDER

- New Uni20 linalg/kernel APIs put API tags and explicit backend selectors
  first, then mutable outputs, then inputs. Examples: `matvec(y, A, x)`,
  `gemm(C, alpha, A, B, beta)`, `gemm(selector, C, alpha, A, B, beta)`,
  and `try_kernel(backend, op, output, inputs...)`.
- Selector-prefix APIs need two overloads when the selector is optional:
  one storage-default overload with no selector, and one constrained
  selector-first overload.
- Older draft examples may still resemble BLAS/LAPACK output-last signatures.
  Treat prefix-tag, output-first ordering as the current design direction unless
  an external ABI boundary forces another order.

## Backend dispatch

### OPERATION-TAG MODEL

- Backend dispatch is an ordered backend-list walk for an operation tag.
- The static capability CPO is
  `consteval KernelTypeAcceptance kernel_accepts_types(backend const&, op const&, args&...)`.
  It checks type-level facts only and returns `no`, `maybe`, or `yes`. It must
  not read argument values; the dispatcher may evaluate it with private
  type-probe lvalues.
- The runtime attempt CPO is `try_kernel(backend, op, args...)`. It performs
  runtime checks such as strides, device placement, handles, and library
  availability, then either runs the kernel and returns `true` or declines
  before side effects and returns `false`.
- A backend that lacks a usable `try_kernel(...)` overload is skipped by
  detection.

### BACKEND VALUES

Current design direction: backend entries are values. They are often stateless
tags, but they may also carry small runtime fields such as device, stream, math
mode, communicator, or placement map.

```cpp
struct CublasBackend {
  int device;
  cudaStream_t stream;
  math_mode_t math_mode;
};

struct CudaGenericBackend {
  int device;
  cudaStream_t stream;
};

struct CpuGenericBackend {};
```

A separate composed selector-state tuple was explored in earlier cytnx-derived
drafts. It is not required for the first Uni20 dispatch layer. If shared backend
state is introduced later, it should be hidden inside the selector value and not
added as a required `State&` parameter to every leaf-kernel CPO.

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
