# Tensor, View, and Backend Dispatch: AI Guidance

- **Audience:** design assistants, coding agents, and reviewers
- **Authority:** non-normative retrieval summary
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-18
- **Canonical sources:** `docs/tensor/operations.md`,
  `docs/architecture/kernel_dispatch.md`, `docs/architecture/overview.md`,
  `AGENTS.md`, current Tensor/linalg source, and focused tests

## Answer rule

- Treat `docs/tensor/operations.md` as canonical for implemented dense Tensor semantics.
- Treat kernel-dispatch documentation as canonical for dispatch contracts.
- Mark future slice, CUDA, distributed, and symmetry-aware designs as unresolved
  unless current source and canonical docs say otherwise.
- Do not revive older draft terminology as current API.

## Layer model

```text
Tensor operation
-> shape, ownership, and output policy
-> storage-derived backend selector
-> resolved mdspan operands
-> operation-value dispatch
-> CPU, BLAS, or LAPACK leaf kernel
```

An Async wrapper adds epoch enrollment and scheduling, then calls the same
synchronous Tensor front end. Leaf kernels do not receive `Tensor` or `Async`.

## Implemented Tensor roles

### `Tensor`

- Concrete owning dense value with compile-time rank.
- Runtime extents are the ordinary default.
- Column-major is the default; named row-major and strided owner aliases exist.
- Ordinary assignment has owner/value replacement semantics, not mdspan rebind semantics.

### `BasicTensor`

- Extents-first alias for a `Tensor` specialization with mixed/static extents.
- It is not a base class or second owner implementation.

### Tensor-view concepts

- `TensorView` requires synchronous extents, `mdspan()`, and backend selection.
- `MutableTensorView` requires a writable resolved mdspan.
- Rank and stridedness are independent refinements.
- These are concepts, not a class hierarchy.
- Tensor-level objects deliberately do not model mdspan concepts directly.

### Resolved mdspan

- Short-lived leaf-kernel operand containing handle, mapping, extents, and accessor.
- It is not a durable owner, async alias, or complete top-level dispatch object.

### Generated and semantic views

- `GeneratedTensor` is readable, compact, and layout-neutral.
- Lazy `conj(tensor)` is implemented and read-only.
- For real tensors, conjugation remains read-only identity semantics.
- Tensor/view construction and `make_tensor(...)` are explicit materialization boundaries.
- Owner-retaining async conjugation and reshape aliases are implemented.

## Operation naming and output policy

- `foo_view(x)`: no-copy alias.
- `foo_inplace(x)`: mutate existing state.
- `assign_foo(out, ...)`: overwrite; old output values do not participate.
- `add_foo(out, ...)`: update; old output values participate.
- `foo(x)` returning an owner: preserve input and allocate/materialize output.
- `foo(std::move(x))`: permission to consume an owning input; reuse is not guaranteed.
- `copy(out, in)` remains the named element-copy operation.
- Do not infer write-through assignment for arbitrary views.

## Accessor semantics

- A pointer-shaped data handle does not prove direct readability/writeability.
- The accessor defines the values observed through `access(...)`.
- A const Tensor-view interface must resolve a const-element mdspan.
- BLAS/LAPACK may bypass accessors only for default access or explicitly recognized
  lowering such as Uni20 conjugation metadata.
- Custom transform, scaling, zip, or proxy accessors require explicit lowering,
  materialization, or a generic accessor-respecting path.
- `uni20::conj` is the project conjugation customization point.

## Backend dispatch

- Dispatch walks an ordered backend list for an operation value.
- Operations may be empty tags or values carrying immutable callable/options state.
- Compile-time type probing and runtime clean decline are distinct.
- Runtime decline must preserve arguments and have no externally visible side effect.
- Once a backend submits work or mutates operands, failure is an operation error,
  not permission to fall back.
- Ordinary fallback must not transfer operands between host, device, or MPI domains.
- Dynamic dispatch is for runtime-erased boundaries such as Python/plugin interfaces,
  not a replacement for the normal static contract.

## Implemented operation surface

Current canonical docs report a substantial dense surface including:

- accessor-aware copy and variadic elementwise overwrite/update;
- reductions, inner products, and norms;
- GEMM/GEMV and matrix initialization;
- matrix exponential;
- exact and truncating SVD;
- self-adjoint and nonsymmetric eigensystems;
- Schur and tridiagonal eigensystem operations.

Backend/scalar/Async coverage is operation-specific. Never infer uniform support.

## Async rules

- Async Tensor wrappers own synchronization and lifetime; backends remain synchronous.
- All caller Tensor operands are `Async<T>`.
- Update outputs use one writer and are not duplicated as inputs.
- Owner-retaining aliases share their parent's exact queue.
- Queue identity catches only obvious aliasing. Arbitrary overlapping storage still
  needs an operation-level contract.

## Open design areas

- General slicing and a concrete non-owning write-through Tensor ref.
- CUDA and distributed Tensor storage/execution.
- Complete symmetry-aware `BlockTensor` lowering.
- General expression/fusion.
- Python Tensor/view ownership and exchange protocols.

Do not present `TensorRef`, backend-state helper types, or old tuple/type-list
composition ideas as implemented unless source inspection confirms them.
