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

- The first synchronous dense implementation is concept-based rather than
  inheritance-based.
- `BasicTensor` owns storage by composition and models `TensorView` and
  `MutableTensorView`.
- `BasicTensor<Element, Extents, StoragePolicy, LayoutPolicy, AccessorFactory>`
  is the configurable extents-based owner. `Tensor<T, Rank>` is the ordinary
  alias with runtime extents on every axis; `DenseMatrix<T>` is its rank-two
  host alias.
- `make_tensor(view)` infers an owning host `Tensor` and deliberately does not
  preserve static input extents.
- `conj(tensor)` is an implemented read-only lazy Tensor view. `copy(...)` and
  `make_tensor(...)` are the explicit eager boundaries.
- There is currently no general concrete non-owning tensor adaptor. Add one only
  with explicit slice/external-storage lifetime and assignment semantics.

## Tensor roles

### Tensor

- `ROLE`: Owning dense tensor value.
- `ASSIGNMENT`: Value/replace semantics.
- `OUTPUT`: Allocation/reallocation policy belongs to operations that explicitly
  take or return an owning `Tensor`.
- `DO NOT CLAIM`: Do not claim `Tensor` assignment is mdspan-style rebind.

### TensorRef

- `ROLE`: Proposed non-owning write-through tensor lvalue, for slices or block
  outputs.
- `ASSIGNMENT`: Write-through into referenced storage when shape is compatible.
- `STATUS`: Proposed / design draft.

### resolved mdspan-like view

- `ROLE`: Leaf-kernel argument: data handle, extents, strides, accessor.
- `ASSIGNMENT`: mdspan-style descriptor rebind.
- `INVARIANT`: A resolved view is not enough for top-level Uni20 dispatch.

### TensorView Concepts

- `ROLE`: Implemented concepts for synchronous dense tensor operands.
- `TensorView`: requires synchronous `extents()`/`extent(axis)` metadata, an
  addressable `mdspan()`, and `backend_selector()`.
- `MutableTensorView`: refines `TensorView` when `mdspan()` is writable.
- `StridedTensorView` / `MutableStridedTensorView`: require affine strided
  resolved spans for providers such as BLAS/LAPACK.
- `RankedTensorView<T, Rank>` and `MutableRankedTensorView<T, Rank>` constrain
  rank independently of stridedness.
- `RankedStridedTensorView<T, Rank>` and its mutable refinement combine these
  properties where a provider boundary needs both.
- `INVARIANT`: Owning tensors and non-owning adaptors may both model these
  concepts; neither must inherit from a common base class.
- `INVARIANT`: A Tensor-view object is not mdspan-like. Its returned mdspan is
  the leaf-kernel operand.

## Candidate tensor concepts

### DESIGN DIRECTION

Top-level synchronous dense operations use the Tensor-view concept family.
Operations with a fixed rank constrain it directly; GEMM uses rank two.
Allocation is deliberately outside these concepts: fixed updates accept mutable
Tensor views, while allocating/value-producing operations take or return a
concrete owning tensor.

### IMPORTANT DISTINCTION

- Leaf kernels use `SpanLike` / mdspan-like resolved views.
- Raw linalg overloads constrain their operands with the weakest applicable
  ranked readable/writable span concepts. GEMM accepts non-strided addressable
  spans because the generic CPU backend supports them; BLAS lowering separately
  requires strided spans.
- Front-end Uni20 dispatch needs a storage-derived backend selector.
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
- Uni20 uses `uni20::conj(span)` and `uni20::conj(tensor)` for lazy conjugating
  views. The Tensor adaptor resolves to the same mdspan accessor. This follows
  the C++26 `std::linalg::conjugated_accessor` model, but Uni20 does not adopt
  `conj-if-needed`; `uni20::conj` is the project-level fix for real-scalar
  conjugation semantics.

### PARAMETER ORDER

- New Uni20 linalg/kernel APIs put API tags and explicit backend selectors
  first, then mutable outputs, then inputs. Examples: `matvec(y, A, x)`,
  `gemm(C, alpha, A, B, beta)`, `gemm(selector, C, alpha, A, B, beta)`,
  `dispatch_kernel(selector, op, output_mdspan, inputs...)`, and
  `try_kernel(backend, op, output, inputs...)`.
- Selector-prefix APIs need two overloads when the selector is optional:
  one storage-default overload with no selector, and one constrained
  selector-first overload.
- Older draft examples may still resemble BLAS/LAPACK output-last signatures.
  Treat prefix-tag, output-first ordering as the current design direction unless
  an external ABI boundary forces another order.

## Backend dispatch

### OPERATION-TAG MODEL

- Backend dispatch is an ordered backend-list walk for an operation tag.
- The concrete linalg GEMM mdspan slice uses generic dispatch with explicit selectors:
  `try_kernel(BlasBackend, gemm_op, ...)` delegates to
  `uni20::linalg::blas::try_gemm(...)`, then falls through to the
  `CpuReferenceBackend` GEMM oracle when BLAS declines.
- Fixed-storage Tensor GEMM is also implemented. `VectorStorage` supplies
  `[LapackBackend, BlasBackend, CpuReferenceBackend]` when BLAS is configured,
  and `[LapackBackend, CpuReferenceBackend]` otherwise. Ineligible operation
  backends are skipped at compile time. Explicit selectors override that
  default.
- `copy_op` has an accessor-respecting CPU reference backend. It is the common
  operation used by `copy` and `make_tensor`; future rank-two BLAS copy
  extensions can accept the same operation by lowering layout and conjugating
  accessor metadata.
- The static capability CPO is
  `consteval auto kernel_accepts_types(backend const&, op const&, args&...)`.
  It checks type-level facts only and returns `kernel_types_no`,
  `kernel_types_maybe`, or `kernel_types_yes`. The dispatcher inspects the
  result type with `decltype` and `std::declval`; it does not construct or read
  argument objects.
- The runtime attempt CPO is `try_kernel(backend, op, args...)`. It performs
  runtime checks such as strides, device placement, handles, and library
  availability, then returns `KernelAttempt::success` or a structured clean
  decline reason such as `unsupported_layout` or `unavailable`.
- A non-success `KernelAttempt` has a strong decline guarantee. The backend must
  preserve every argument and must not mutate operands, submit work, commit
  storage, or produce another externally visible side effect. The dispatcher
  invokes candidates with stable lvalue arguments and does not copy operands or
  mdspan descriptors to conceal contract violations. Once work starts, failure
  is an operation error rather than fallback. A backend returning
  `kernel_types_yes` must return `KernelAttempt::success`.
- Terminal provider failures throw, abort through a logic check, or appear in
  an operation-specific result. They are not `KernelAttempt` values and must
  never trigger fallback.
- `kernel_accepts_types(...)` is the mandatory type gate and may be narrowly
  constrained. If it is not callable for the exact argument types, acceptance
  is a hard `no`, even when `try_kernel(...)` is broadly callable.
- Generic code may use
  `probe_dispatch_kernel(backends, op, args...)`. It inspects only deduced types
  and aggregates the candidate list: any `yes` gives `yes`, otherwise any
  `maybe` gives `maybe`, otherwise `no`. A non-callable backend type gate
  contributes `no`. The safe single-backend query is an implementation detail.
- Use `try_dispatch_kernel(...)` when exhausting all runtime candidates is an
  expected result that the caller will handle. Use `dispatch_kernel(...)` for
  checked execution; it raises structured `KernelDispatchError`, which the
  presentation layer renders before aborting in native C++ and which propagates
  as an exception after Python module initialization. Both normal C++ entry
  points are constrained out when the aggregate type probe is `no`, preserving
  compile-time diagnosis.
- Use `dynamic_dispatch_kernel(...)` only at Python, plugin, or runtime-erased
  boundaries that must remain callable for a statically unavailable kernel. It
  converts both a type-level `no` and runtime backend exhaustion into
  `KernelDispatchError`.
- Keep the matching `try_kernel(...)` broadly callable instead of repeating its
  type test in a long `requires` clause. Dispatch is the contract boundary:
  `try_kernel(...)` may assume the type gate accepted. Do not add a redundant
  `static_assert` merely to diagnose unsupported direct calls; backend
  `try_kernel` functions are not direct APIs.
- A backend that lacks a usable `try_kernel(...)` overload is skipped by
  detection.
- Keep storage-default backend lists in one storage and execution domain. Host
  selectors may contain BLAS and CPU reference backends; future CUDA selectors
  contain only CUDA-device backends. Ordinary decline must not copy operands to
  another domain. Any emergency device-to-host route is an explicit
  operation-specific composite kernel or higher-level policy.

### BACKEND VALUES

Backend entries are values, but default storage selectors should normally be
stateless candidate lists. Operand location is not selector state: memory kind
belongs to the accessor/handle type, and runtime location such as a CUDA device
belongs to the accessor-defined data handle. This keeps transformed and sliced
views from duplicating location state.

```cpp
struct CublasBackend {};
struct CudaGenericBackend {};

struct CpuReferenceBackend {};
```

Stateful selector entries remain an explicit override mechanism for genuine
operation context or options, such as a selected stream, MPI communicator,
workspace policy, math mode, or multiprecision setting. They should not become a
second source of truth for operand location.

## Temporaries

### DESIGN DIRECTION

Temporary allocation is separate from computation.

- The current host API is `make_tensor(view)` or
  `make_tensor<Layout>(selector, mdspan)`. The latter requires an explicit
  selector because a bare mdspan does not carry storage-domain policy. Both
  forms return a fixed-rank `Tensor` with runtime extents on every axis.
- Operand-temporary type/storage comes from an owning storage domain or explicit
  allocator/factory. A stateful selector override may contribute options but is
  not the primary owner of operand location.
- Filling an operand temporary is an ordinary copy/evaluation kernel.
- Direct mdspan entry points that need operand temporaries must provide explicit
  backend selector state and storage-domain information.
- LAPACK work arrays are not operand materialization. A direct LAPACK operation
  wrapper may allocate `work`/`rwork`/integer work arrays after a workspace query
  while still being direct for its matrix/vector operands.
- Copying, packing, transposing, or conjugating a user-visible operand into
  scratch storage is operand materialization. That belongs in a prepared wrapper
  or an explicitly documented higher-level API, not in a silent direct wrapper.

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

- `TensorView` is a concept, not a concrete base class.
- A resolved mdspan is a leaf-kernel operand, not a tensor-level owner or durable
  asynchronous alias.
- Backend fallback order should not be encoded by inheritance.
- A CUDA backend tag by itself is not enough to allocate CUDA temporary storage.
- `std::tuple` can store duplicate types, but `std::get<T>` by type requires
  `T` to be unique.
- `std::type_list` does not exist in the C++ standard library.
