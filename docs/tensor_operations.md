# Tensor Operations, Semantics, and Async Support

**Status:** canonical guide for the currently implemented dense Tensor
operation model. This page records behavior that exists in the codebase and
separately identifies operations that do not yet have an Async wrapper.

This guide covers local dense tensors, tensor-level views, and their lowering
to dense kernels. Symmetry-aware `BlockTensor` operations have additional
metadata and selection-rule contracts and must not use an implicit dense
fallback.

## Header Map

The main entry points are:

```cpp
#include <uni20/tensor/tensor.hpp>       // Tensor, DenseMatrix, generated values, views, reshape
#include <uni20/tensor/copy.hpp>         // copy, make_tensor, materializing reshape
#include <uni20/tensor/async.hpp>        // async::conj
#include <uni20/linalg/linalg.hpp>       // synchronous dense linalg operations
#include <uni20/linalg/async.hpp>        // implemented Async linalg wrappers
```

Including a narrower operation header is preferred in library implementation
files when it keeps dependencies clear.

## Layer Model

Tensor operations have three distinct layers:

```text
Tensor operation
  -> shape, ownership, and output policy
  -> storage-derived backend selector
  -> resolved mdspan operands
  -> backend dispatch
  -> leaf kernel
```

An Async operation adds scheduling around the same synchronous operation:

```text
Async<Tensor> operands
  -> enroll ReadBuffer and WriteBuffer epochs
  -> scheduled coroutine awaits stored Tensor values
  -> synchronous Tensor operation
  -> mdspan dispatch and leaf kernel
```

Leaf kernels do not receive `Tensor` or `Async<Tensor>`. They receive resolved
mdspan-like operands after ownership, storage domain, output policy, and async
ordering have been handled by higher layers.

### Type Roles

| Type or concept | Role |
|---|---|
| `BasicTensor<Element, Extents, ...>` | Configurable owning tensor with compile-time rank and possibly static extents. |
| `Tensor<Element, Rank, ...>` | Ordinary owner with compile-time rank and runtime extents on every axis. |
| `DenseMatrix<Element, Layout>` | Rank-two host `Tensor`; column-major by default. |
| `GeneratedTensor` | Compact read-only tensor whose accessor computes values without dense element storage. |
| `TensorView` | Readable tensor-level object exposing extents, `mdspan()`, and a backend selector. It is a concept, not a base class. |
| `MutableTensorView` | `TensorView` whose resolved mdspan permits writes. |
| `OwningTensor` | Explicit opt-in classification whose move operation transfers the storage and lifetime exposed by `mdspan()`. |
| `StridedTensorView` | Tensor view whose resolved mdspan has a strided mapping. Operations such as reshape may require this refinement. |
| Resolved mdspan | Short-lived leaf-kernel operand containing handle, mapping, extents, and accessor semantics. |

Tensor objects deliberately do not model mdspan concepts. Front-end operations
accept tensor concepts and resolve `.mdspan()` only after any shape-changing
work is complete.

## Operation Vocabulary

Names communicate ownership and mutation. New operations should use these
forms unless the underlying mathematical convention makes another name
materially clearer.

| Form | Semantic contract |
|---|---|
| `foo_view(x)` | Return a no-copy alias. The result does not extend an ordinary synchronous source lifetime unless its type explicitly says otherwise. |
| `foo_inplace(x)` | Mutate the existing tensor, its elements, or its descriptor. No independent value is returned. |
| `assign_foo(out, ...)` | Overwrite `out`. A resizable output may replace its shape and allocation. Old output values do not participate. |
| `add_foo(out, ...)` | Compound update. Existing output values participate, so shape is fixed and the output must already exist. |
| `foo(x)` returning an owner | Preserve lvalue inputs and allocate or materialize an owning result. |
| `foo(std::move(x))` | Grant permission to consume an owning input and transfer its allocation. Reuse is not guaranteed. |
| `make_tensor(view)` | Explicit eager materialization boundary for a view or generated expression. |

There is no general rule that assignment to an arbitrary tensor view means
element assignment. A plain mdspan assignment rebinds a descriptor. A future
write-through slice proxy must define that behavior explicitly.

### Assignment by Object Category

| Object | Meaning of ordinary assignment |
|---|---|
| Owning `Tensor` or `BasicTensor` | Replace the owned value and descriptor state through the concrete type's copy/move assignment. Storage may be reused or replaced. |
| Plain mdspan | Rebind the mdspan descriptor. Elements are not copied. |
| Read-only semantic view | Assignment through the view is unavailable. |
| Future write-through slice/ref | Assign elements into the referenced region according to an explicitly documented contract. |

Elementwise copying remains the named `copy(out, in)` operation. This avoids
making descriptor assignment depend on whether a particular view happens to
own storage.

## Core Semantic Rules

### Readable, Mutable, and Owning Are Independent

A type may own compact descriptor state without owning a writable dense
allocation. `GeneratedTensor` is readable but neither mutable nor an
`OwningTensor`. A lazy conjugating view is also readable and non-owning.

Only a mutable owning rvalue is eligible for storage transfer. Passing a view
by value does not make it consumable.

### Views and Materialization

Structural views change addressing while preserving the direct write law.
Examples include reshape and future slice or component views. They may be
writable when their accessor and source permit it.

Future `real(x)` and `imag(x)` views of non-const complex storage are structural
component slices: they change scalar type, handle, and stride and may be
writable. They are not writable proxy-accessor transformations. Generic mutable
operation outputs should normally expose direct/default-access semantics;
writable proxy accessors require an operation-specific assignment law and
backend lowering.

Semantic transform views change values observed through an accessor. Complex
`conj(x)` is therefore read-only. For real tensors, `conj(x)` returns a const
identity view. Applying `conj` twice returns a read-only view of the original
values rather than a writable left-hand side.

Materialization is explicit:

```cpp
auto lazy = uni20::conj(matrix);
auto owned = uni20::make_tensor(lazy);

uni20::Tensor<uni20::complex<double>, 2> output;
uni20::copy(output, lazy);
```

`copy` observes accessor semantics. A raw pointer-shaped data handle is not
enough to bypass an accessor; a provider backend may do so only for a default
accessor or a recognized accessor whose semantics it lowers explicitly.
A bare mdspan carries no storage policy, so its `copy` or `make_tensor`
operation requires an explicit backend selector. Tensor-level views supply
their selector through storage policy.

### Output Shape

Output operations use two contracts:

```cpp
uni20::require_shape(output, required); // validate only
uni20::ensure_shape(output, required);  // resize owner or validate fixed view
```

An update operation uses `require_shape` because old output values participate.
An overwrite operation may use `ensure_shape`. A resizable owner retains a
matching allocation and calls `reset_shape` only when the shape differs. A
fixed view can never resize.

Shape preparation must happen before resolving the output mdspan because
resizing may invalidate its handle.

### Move and Storage Reuse

`std::move` grants permission; it does not promise reuse. The operation may
still materialize when layout, accessor, storage policy, scalar type, or backend
requirements are incompatible.

Current examples are:

```cpp
auto reshaped = uni20::reshape(std::move(tensor), 6, 4);
auto eigensystem = uni20::linalg::eigh(std::move(matrix));
```

Existing synchronous views into transferred storage are invalidated under
ordinary C++ moved-from owner rules. Uni20 does not track them dynamically.

### Aliasing

Unless an operation explicitly documents in-place behavior, mutable output
storage must not overlap an input. `assign_product` and `add_product` cheaply
reject the obvious same-object case, but they do not attempt a complete strided
range-overlap proof. C++ callers remain responsible for less obvious overlap.

Async matrix product rejects an output that shares its exact `EpochQueue` with
either input before enrolling buffers. Such an operation would otherwise
self-block. Read-only inputs may share a queue.

### Backend Selection

Default backend selection comes from storage policy and operation type. An
explicit-selector overload is available for backend-sensitive code and tests.
Concrete backends may still decline after mdspans reveal unsupported layout or
accessor details.

`GeneratedStorage` is backend-neutral. In a mixed operation, concrete storage
selects the backend list. With only generated operands, the host reference
backend is the fallback. Each kernel still decides whether it understands the
generated accessor.

## Core Tensor Operation Support

In the tables below, "Not implemented" means there is no public overload that
accepts `Async<Tensor>` operands. A developer can still await values in an
operation-specific coroutine and call the synchronous API, but that coroutine
must implement the ownership, buffer, aliasing, and failure contracts described
later in this guide.

| Operation | Synchronous semantics | Allocation or alias behavior | Dedicated Async support |
|---|---|---|---|
| `full`, `zeros`, `ones`, `eye` | Create readable generated tensors. | Compact generator state; no dense element allocation. | No Async-specific overload. |
| `conj(x)` | Read-only lazy semantic view. | Aliases the source; no copy. | `async::conj(x)` returns an owner-retaining alias on the same epoch queue. |
| `reshape_view(x, ...)` | No-copy structural reshape of a canonical contiguous mapping. | Aliases an lvalue source and preserves accessor semantics. | Not implemented. |
| `reshape_inplace(x, ...)` | Replace an owning tensor mapping at the same compile-time rank. | Keeps the allocation; existing copied descriptors keep their old mapping. | Not implemented. |
| `reshape(x, ...)` | Return an owning reshaped value. | Lvalue copies; owning rvalue may transfer; non-owning strided input materializes. | Not implemented. |
| `copy(out, in)` | Overwrite while observing input accessor semantics. | Resizes a resizable output or validates a fixed output. | Not implemented. |
| `make_tensor(view)` | Materialize a readable tensor or mdspan as an owning host tensor. | Allocates an inferred runtime-extents owner. | Not implemented. |
| `conjugate_inplace(x)` | Eager element mutation. Real values are unchanged. | Reuses existing storage. | Not implemented. |
| `require_shape`, `ensure_shape` | Output-policy helpers for operation authors. | Validate only, or resize when the output type permits it. | Used inside wrappers; not standalone Async operations. |

For a rank-`R` result, generalized `eye<T>(n0, ..., n{R-1})` returns
one exactly when all indices agree. It includes scalar one, an all-ones
rank-one tensor, rectangular identity matrices, and higher-rank tensor-network
copy tensors.

The reshape contracts, including `-1` inference and contiguous-order rules,
are detailed in [Generated Tensors and Reshape](tensor_creation_and_reshape.md).

## Dense Linear Algebra Support

These are tensor-front-end operations even though their leaf kernels receive
mdspans or spans.

| Operation | Synchronous contract | Output/storage behavior | Async support |
|---|---|---|---|
| `gemm` | Fixed-output `C = alpha * A * B + beta * C`. | Caller supplies compatible output; no resize. | No direct wrapper. Use `assign_product` or `add_product` for the common overwrite/update forms. |
| `gemv` | Fixed-output `y = alpha * A * x + beta * y`. | Caller supplies compatible output; no resize. | Not implemented. |
| `assign_product` | Overwrite matrix product. | Output may resize; old values are ignored. | Implemented for all-Async tensor operands; `alpha` may be immediate or Async. |
| `add_product` | Accumulate `output += alpha * lhs * rhs`. | Output must exist and have the required shape. | Implemented for all-Async tensor operands; `alpha` may be immediate or Async. |
| `set_matrix` | Set diagonal/off-diagonal values in a selected matrix region. | In-place mutation. | Not implemented. |
| `matrix_exponential` | Compute into a fixed rank-two output. | Caller supplies compatible output. | Not implemented. |
| `self_adjoint_eigh` | Destructive LAPACK-style workspace operation. | Matrix workspace is overwritten; eigenvalue output may resize. | No direct wrapper. |
| `eigh(matrix)` | Preserving value operation returning eigenvalues and eigenvectors. | Materializes work storage and returns two owners. | Implemented; returns two independent Async outputs. |
| `eigh(std::move(matrix))` | Consuming value operation. | May transfer a compatible owning allocation to eigenvectors. | Implemented for `Async<OwningTensor>&&`; consumes the stored value on success. |
| `nonsymmetric_eigen` | LAPACK-style nonsymmetric eigensystem. | Destructive matrix workspace and caller-provided outputs. | Not implemented. |
| `schur`, `hessenberg_schur` | LAPACK-style decomposition. | Destructive matrix workspace and caller-provided outputs. | Not implemented. |
| `reorder_schur` | Reorder an existing Schur form. | In-place mutation of Schur form and optionally vectors. | Not implemented. |
| `symmetric_tridiagonal_eigen` | Tridiagonal eigensystem over caller-provided spans and tensor output. | LAPACK-style mutable work/output buffers. | Not implemented. |

The fixed-output `gemm` and `gemv` forms are low-level tensor front ends. New
ordinary application code should prefer operation-specific overwrite, update,
or value-returning APIs where those exist because their output policy is
explicit.

## Async Tensor Contract

### All Tensor Operands Are Async

An Async overload does not mix synchronous and asynchronous tensor operands:

```cpp
uni20::async::Async<uni20::DenseMatrix<double>> lhs = /* ... */;
uni20::async::Async<uni20::DenseMatrix<double>> rhs = /* ... */;
uni20::async::Async<uni20::DenseMatrix<double>> output;

uni20::linalg::assign_product(output, lhs, rhs);
```

Ordinary scalar/configuration values are passed by value. An operation may
explicitly allow an `Async<Scalar>`; matrix-product `alpha` does. Borrowed
objects such as spans and string views remain borrowed even when their
descriptor is passed by value and require separate lifetime analysis.

### Async Is a Scheduling Wrapper

The wrapper resolves the immutable selector from tensor/storage types before
scheduling, enrolls buffers, and moves all coroutine state into a named
coroutine or captureless static coroutine lambda. The coroutine awaits stored
Tensor values and invokes the synchronous operation. Runtime backend declines
still happen after the mdspans are available.

Do not make `Async<Tensor>` model `TensorView`, and do not add Async awareness
to leaf backends.

### Overwrite and Update Outputs

For async `assign_product`, an unconstructed output may be constructed from the
required extents. An existing output follows synchronous `ensure_shape`
semantics.

For async `add_product`, output is both input and output. It uses one
`WriteBuffer`, must already be constructed, and cannot resize. Taking separate
read and write buffers for the same timeline is incorrect.

Ordinary async write-proxy assignment follows the stored type. The first write
constructs uninitialized storage when possible; later writes invoke the stored
type's assignment operator. It does not silently reinterpret mdspan assignment
as elementwise copy. Use explicit `emplace` when reconstruction is required and
a named tensor operation when backend-dispatched element evaluation is
required.

### Async Views and Lifetime

`async::conj(parent)` returns `Async<ConjugatedTensorView<...>>` for complex
tensors and an async const identity view for real tensors. The alias owns its
descriptor, retains the parent storage control block, and shares the parent's
exact `EpochQueue`.

A future async structural view such as reshape or slice must use the same
owner-retaining alias mechanism. Wrapping a raw synchronous view in a fresh
`Async` would create an independent queue for aliased bytes and is incorrect.

### Multiple Outputs and Consuming Inputs

Async `eigh` returns independent handles:

```cpp
auto [eigenvalues, eigenvectors] = uni20::linalg::eigh(matrix);
```

This avoids an extraction coroutine and allows either result to flow directly
into downstream work. Every output writer is an exception sink.

The consuming overload accepts `Async<OwningTensor>&&`, enrolls the input as a
writer, and removes its stored value with the consuming buffer API. On success,
the input timeline no longer contains a readable matrix. The writer remains
gated until both outputs are committed so a failure can reach the input and
both output epochs consistently.

### Failures

An early backend decline is not a failure and must have no externally visible
side effects. Once a synchronous operation fails to produce its result, the
async task fails rather than trying another backend.

Unhandled coroutine exceptions propagate to every output `WriteBuffer` passed
to the task. Reads of those output epochs observe the failure. Synchronous
`CHECK` and `PANIC` still identify internal logic errors and may terminate the
process; bindings must validate user input before entering such paths.

## Adding an Async Operation

Before adding an Async overload:

1. Implement and test the synchronous Tensor operation first.
2. Classify every output as overwrite, update, allocating, or consuming.
3. Decide whether outputs are one `Async<Result>` or several independent Async
   values. Prefer independent handles when downstream operations use results
   separately.
4. Resolve the default selector from Tensor/storage types before scheduling.
5. Reject obvious output/input queue aliasing before buffer enrollment.
6. Use one `WriteBuffer` for each mutated timeline and `ReadBuffer` for each
   read-only Tensor input.
7. Pass selectors, options, and other ordinary state into the coroutine by
   value. Never use a capturing coroutine lambda; scheduled coroutine lambdas
   must be `static`.
8. Await Tensor values before resolving mdspans, then call the synchronous
   Tensor operation.
9. Ensure every output writer receives task failures.
10. Test pending-input lifetime, unconstructed outputs, aliases, numerical
    behavior, and exception propagation.

Keep wrappers operation-specific until several more implementations establish
a genuinely reusable output and lifetime pattern.

## Current Gaps

The following are intentionally not implied by the current API:

- no Async `copy` or `make_tensor`
- no Async reshape or general slice alias
- no Async `gemv`, matrix exponential, or general LAPACK workspace operation
- no allocating value API for most destructive LAPACK front ends
- no general concrete synchronous `TensorRef` slice proxy
- no subrange epoch tracking; async aliases conservatively share a whole parent
  queue
- no implicit dense fallback for symmetry-aware tensors

These are implementation targets, not compatibility promises. New work should
preserve the semantic categories in this document while choosing the clearest
API for the operation.

## Related Documents

- [Generated Tensors and Reshape](tensor_creation_and_reshape.md)
- [Kernel Dispatch Design](kernel_dispatch.md)
- [Backend Dispatch Design](backend_dispatch.md)
- [Async Tensor Kernel Authoring](async/kernel_authoring.md)
- [Async Runtime Model](async/runtime_model.md)
- [Async Buffers and Awaiters](async/buffers_and_awaiters.md)
- [Async Tensor Lifetime and Dispatch Draft](async_tensor_lifetime_and_dispatch_draft.md)
- [Tensor Dispatch and View Semantics Draft](tensor_dispatch_and_view_semantics_draft.md)
