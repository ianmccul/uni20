# Kernel dispatch

This document defines the current Uni20 kernel-dispatch contract. It is normative for
backend selection, type eligibility, runtime decline, fallback, diagnostics, and
coroutine-aware kernel execution.

Provider-specific lowering belongs in the relevant backend documentation. Tensor shape
management and mdspan lowering are described in
[`../linalg/mdspan_dispatch.md`](../linalg/mdspan_dispatch.md). Execution resources and
scheduler behavior are described in [`execution.md`](execution.md).

## The model in one paragraph

Kernel dispatch walks an ordered backend selector for an operation tag.
`kernel_accepts_types` classifies each backend as `no`, `maybe`, or `yes` from the
argument types. A `no` candidate is not instantiated, a `maybe` candidate may cleanly
decline at runtime, and a `yes` candidate must succeed. `try_kernel` performs the
runtime attempt. Every non-success result must preserve inputs and fixed or update
outputs and must occur before work submission, result writes, or any other externally
visible execution effect. An operation-declared replaceable output is the narrow
exception: a backend may provisionally construct, resize, or replace it before
declining, and a later backend may reuse or replace that prepared output.
`dispatch_kernel` reports exhaustion, while `try_dispatch_kernel` returns whether any
candidate accepted the operation. `co_dispatch_kernel` follows the same ordered walk
and may use `try_make_kernel_task` when a backend needs to suspend for resource admission.

## Scope

Kernel dispatch answers one question:

> Which implementation in an ordered backend selector can perform this operation on
> these tensor-view operand types and values?

Except for operation-authorized provisional preparation of a replaceable output, it
does not:

- choose tensor shapes or allocate outputs;
- transfer values between storage domains;
- flatten symmetry-aware values to dense storage;
- infer raw provider access from a pointer-shaped mdspan handle;
- define epoch dependencies for `Async<T>`;
- plan distributed communication.

Those responsibilities belong to higher-level operation wrappers or lower-level
provider adapters.

## Core vocabulary

### Dispatch operand boundary

Each operation tag defines its operand boundary. Fixed existing operands no
longer need tensor policy after the frontend has selected a backend list.
Copy, fixed-output matrix products, elementwise transforms, reductions, matrix
initialization and exponentiation, and fixed-output LAPACK decompositions
therefore enter `probe_dispatch_kernel`, `kernel_type_candidates`,
`try_dispatch_kernel`, `dispatch_kernel`, or `co_dispatch_kernel` as
`MdspecLike` operands, with the applicable mutable and ranked
refinements. Their `kernel_accepts_types`, `try_kernel`, and
`try_make_kernel_task` customizations use the same descriptor types. Scalar
coefficients, axes, host `std::span` work arrays, and other non-tensor
parameters remain ordinary values.

The tensor frontend must select the backend while storage and execution policy
are still available, then normalize each fixed operand exactly once. A readable
input is normalized through the tensor's const interface and a writable output
through its mutable interface. An `ImmediateTensorView` produces an ordinary
mdspan; a descriptor-backed `TensorView` produces concrete `mdspec` metadata.
A bare mdspan convenience API directly supplies the
same normalized descriptor boundary and does not need a temporary tensor
facade.

The frontend materializes each fixed descriptor as a local value before the
backend walk. This copies the mapping, accessor, and data descriptor; it does
not acquire a handle, migrate data, or extend the lifetime of underlying
storage. A descriptor may be borrowed, so its source storage must remain alive
through every backend attempt and any work whose access lease refers to it.
Async frontends retain the applicable epoch storage while descriptor values are
held in the coroutine or kernel-task frame.

Normalization preserves the mathematical object and every resource identity
needed by a selected backend. A descriptor for a future block-sparse,
distributed, file-backed, or remotely staged tensor may therefore carry
symmetry sectors, distribution metadata, communicators, file mappings, or
storage-placement identities in addition to extents, mapping, and accessor
state. Normalization never means dense projection, host materialization, or
storage transfer.

A replaceable output is not a fixed multidimensional descriptor.
`assign_product_op{}` therefore has a heterogeneous dispatch signature. Its
synchronous output is a mutable tensor object and its async output may be an
unconstructed `shared_storage<Tensor>`, while its readable fixed inputs are
normalized before the backend walk. The selected backend chooses and prepares
the output, then normalizes the resulting fixed descriptor.

Functions that operate directly on resolved mdspans sit below operation dispatch.
They are provider/library API calls or lower-level Uni20 module interfaces, not
`xxxx_op{}` dispatch customization points. This separation keeps acquisition,
execution-domain validation, and accessor lowering inside the selected backend.

Within a replaceable-output backend attempt, retain the readable descriptors
received from dispatch. Prepare the output before retaining its writable
descriptor because preparation may invalidate an earlier descriptor. After
lowering, invoke the backend's private descriptor or provider implementation
directly; do not redispatch the resolved operands through another operation-tag
customization.

### Operation tag

An operation tag is a lightweight value identifying a kernel family. Operation tags
must define a stable static `name` used by diagnostics.

Examples include `assign_product_op`, `gemm_op`, `gemv_op`, and
decomposition-specific operation tags.

The operation tag is passed to all backend customization points. This keeps the
dispatcher generic and allows one backend value to implement many operations.

Operation tags also carry output semantics that cannot safely be inferred from
runtime scalar values. `assign_product_op` permits a replaceable output because
the old output value is irrelevant. Its dispatch signature has no `beta`
argument. `gemm_op` treats the output as an existing fixed-storage operand and
never authorizes rebinding, even when `beta` is numerically zero or is represented
by an opaque backend scalar handle. A backend may lower both operations to one
provider GEMM implementation after applying their different output contracts.
Because `assign_product_op` declares its output replaceable, a candidate may prepare
that output before its final layout or transform check. A decline may therefore leave
the output prepared for the next candidate, but may not write result elements.

### Backend

A backend is a named value that participates in dispatch. Backend values must define a
stable static `name`.

A backend may be stateless or may hold configuration and shared runtime state. The
ordered backend selector preserves backend values and their order. Dispatch presents
each backend as a stable lvalue during the walk.

### Backend selector

A selector is either one backend value or a `backend_list<...>`. A single backend is
normalized to a one-entry list.

```cpp
backend_list{
    LapackBackend{},
    BlasBackend{},
    CpuReferenceBackend{},
};
```

Order is semantic. The first eligible backend that succeeds performs the operation.

Tensor operation wrappers normally obtain a selector from the operand storage policies:

```cpp
auto selector = select_backend(operation, output, lhs, rhs);
```

Code that has only tensor types may use the type-based form:

```cpp
auto selector =
    select_backend_for<OutputTensor, LhsTensor, RhsTensor>(operation);
```

Selector resolution is based on static tensor and storage-policy information. It does
not inspect tensor values. Backend-neutral storage policies may participate with a
backend-bound storage policy, but incompatible backend-bound storage policies require
an explicit selector or an explicit transfer.

### Selector resolution

Default Tensor operations resolve selectors in the following order:

| Priority | Mechanism | Meaning |
|---|---|---|
| 1 | Explicit selector argument | Per-call selection; the default-selection mechanism is bypassed. |
| 2 | `backend_selector_override<Operation, StoragePolicy>` | User replacement for one operation and storage-policy combination. |
| 3 | `backend_selector_default<Operation, StoragePolicy>` | Uni20-owned operation-specific composition of the storage selector. |
| 4 | `StoragePolicy::backend_selector()` | General execution capabilities supplied by the storage policy. |

A library operation-specific default defines:

```cpp
static auto select(Operation const& operation,
                   StorageSelector storage_selector);
```

The storage selector is passed by value, so the default may retain it inside
stateful operation-specific backends. This supports orthogonal policy layers.
For example, a tensor-contraction default can return Direct-GEMM, Looped-GEMM,
and Packed-GEMM contraction backends that each retain the BLAS/CPU or
cuBLAS/CUDA selector supplied by storage.

`backend_selector_default` is a Uni20 library customization. A user who needs
to replace the selected list specializes `backend_selector_override` instead:

```cpp
template<>
struct backend_selector_override<my_operation, my_storage_policy>
{
    static auto select(my_operation const&)
    {
      return backend_list{KnownGoodBackend{}};
    }
};
```

The override is a complete replacement. Uni20 does not prepend or append the
library default, and exhaustion does not fall back to it. This permits a user
to remove a faulty or unsuitable kernel immediately without modifying Uni20.
An explicit selector passed to an operation has still higher priority because
the operation does not call `select_backend` in that overload.

## Backend customization points

A backend participates in an operation through free functions found by ordinary
overload resolution and argument-dependent lookup.

### Type eligibility

```cpp
kernel_accepts_types(backend, operation, args...)
```

The function is evaluated at compile time and must return one of:

```cpp
kernel_types_no
kernel_types_maybe
kernel_types_yes
```

The dispatcher probes argument types as unevaluated stable lvalues. Reference
qualification is removed and then restored as an lvalue reference; cv-qualification is
preserved. The probe therefore matches the lvalue argument form used during the runtime
walk without evaluating or copying values.

If no matching `kernel_accepts_types` customization exists, the result is a hard
`kernel_types_no`.

A backend returning `maybe` or `yes` must provide a matching `try_kernel`
customization. The dispatcher checks that contract at compile time.

#### `no`

`no` means that this backend does not implement the operation for these argument types.
The runtime implementation is not instantiated for this candidate.

Use `no` for static incompatibilities such as:

- unsupported scalar or element types;
- incompatible rank;
- an output type that is not writable;
- an accessor or handle category the backend cannot lower;
- a kernel family not implemented by the backend.

#### `maybe`

`maybe` means that the argument types are supported, but runtime values determine
whether this backend can perform this instance.

Typical runtime checks include:

- shape;
- stride and layout representation;
- transform support;
- aliasing;
- device identity;
- provider availability;
- bounded execution-resource admission.

A `maybe` backend may return any non-success `KernelAttempt`, subject to the clean
decline contract.

#### `yes`

`yes` means that every valid runtime instance of the admitted argument types is handled
by this backend. Its `try_kernel` call must return `KernelAttempt::success`.

The dispatcher checks this invariant. Do not use `yes` merely because a backend is the
last configured fallback. Use it only when the implementation is total over its
admitted type domain.

#### Expressing the accepted type domain

`kernel_accepts_types` may state its constraints directly or probe a constrained
lower-level implementation function:

```cpp
if constexpr (requires {
    {
        detail::try_gemm(output, alpha, lhs, rhs, beta)
    } -> std::same_as<KernelAttempt>;
}) {
    return kernel_types_maybe;
} else {
    return kernel_types_no;
}
```

This is useful when the constrained lower-level function is the single source of truth
for the backend's static domain.

Do not infer support by probing an unconstrained function template whose body contains
the real requirements. Callability does not instantiate and validate the function
body. In that case, put the constraints on the probed function or state them directly
in `kernel_accepts_types`.

The synchronous and coroutine implementations of a backend must admit the same static
domain unless the operation contract explicitly distinguishes them.

### Runtime attempt

```cpp
try_kernel(backend, operation, args...) -> KernelAttempt
```

`try_kernel` receives the same stable lvalue arguments used by the dispatch walk. It
returns one of:

```cpp
KernelAttempt::success
KernelAttempt::unsupported_instance
KernelAttempt::unsupported_shape
KernelAttempt::unsupported_layout
KernelAttempt::unsupported_accessor
KernelAttempt::unsupported_transform
KernelAttempt::incompatible_devices
KernelAttempt::unavailable
KernelAttempt::insufficient_resources
```

Every value other than `success` is a clean decline. A decline must preserve inputs
and fixed or update outputs, submit no work, and publish no result. When the operation
declares an output replaceable, its contract may permit provisional construction,
resizing, or replacement before a decline. Later candidates receive that prepared
output and may reuse or replace it.

Terminal failures are not represented by `KernelAttempt`. Once execution is committed,
provider failures, task failures, and operation failures are reported through the
operation's ordinary error or exception mechanism.

### Optional coroutine attempt

A backend that needs suspendable resource admission may additionally define:

```cpp
try_make_kernel_task(backend, operation, args...)
    -> KernelTaskAttempt<ConcreteTask>
```

`ConcreteTask` must derive from `async::BasicTask`.

`try_make_kernel_task` is an ordinary fallible task factory, not a coroutine.
Any coroutine it creates uses a `co_`-prefixed name.

`KernelTaskAttempt` represents one of three states:

1. **Decline:** a non-success `KernelAttempt` and no task.
2. **Completed success:** `success` and no task.
3. **Deferred success:** `success` and a task.

A decline has the same input-preservation and provisional replaceable-output
preparation contract as an ordinary `try_kernel` decline.
A completed success represents an operation that is already finished, such as a
zero-size output. A deferred success commits the backend when the task is awaited;
failures after that point are terminal and must not trigger fallback.

Backends without `try_make_kernel_task` are invoked through their ordinary `try_kernel`
implementation on the current scheduler thread.

## Type probing and candidate filtering

The dispatch layer provides three related facilities.

### `probe_dispatch_kernel`

```cpp
auto acceptance = probe_dispatch_kernel(selector, operation, args...);
```

This returns the aggregate type-level result for the selector:

- `yes` if at least one backend reports `yes`;
- otherwise `maybe` if at least one backend reports `maybe`;
- otherwise `no`.

No runtime values or backend state are inspected.

### `kernel_type_candidates_t`

```cpp
using candidates =
    kernel_type_candidates_t<Selector, Operation, Args...>;
```

This is the ordered `backend_list` type obtained by removing every backend whose
type-level result is `no`.

### `kernel_type_candidates`

```cpp
auto candidates =
    kernel_type_candidates(selector, operation, args...);
```

This returns the corresponding backend values. Candidate order and backend state are
preserved.

These facilities are useful for compile-time conformance tests and for code that needs
to inspect or exercise each statically eligible backend independently.

## Runtime dispatch front ends

### `try_dispatch_kernel`

```cpp
bool succeeded =
    try_dispatch_kernel(selector, operation, args...);
```

This function is available when at least one backend accepts the argument types. It
walks candidates in selector order and returns:

- `true` when a backend succeeds;
- `false` when every eligible backend cleanly declines.

A `yes` candidate must succeed and therefore terminates the walk.
On `false`, inputs and fixed outputs are preserved; replaceable outputs remain valid
but may contain provisional backend preparation. This postcondition does not require
restoring a replaceable output's previous allocation. Its element values are
unspecified.

### `dispatch_kernel`

```cpp
dispatch_kernel(selector, operation, args...);
```

This function performs the same ordered walk but reports
`KernelDispatchFailure::all_candidates_declined` if no candidate succeeds.

It is constrained to calls with at least one type-eligible backend. This gives normal
C++ callers an early compile-time error when the configured selector cannot implement
the operation for the argument types.

### `dynamic_dispatch_kernel`

```cpp
dynamic_dispatch_kernel(selector, operation, args...);
```

This boundary remains callable even when no backend accepts the concrete argument
types. It is intended for runtime-erased interfaces such as Python bindings.

It reports:

- `KernelDispatchFailure::no_eligible_backend` when all type-level results are `no`;
- `KernelDispatchFailure::all_candidates_declined` when candidates exist but all
  decline at runtime.

Do not use `dynamic_dispatch_kernel` merely to avoid satisfying the static dispatch
contract in ordinary C++ code.

## Coroutine-aware dispatch

```cpp
co_await co_dispatch_kernel(selector, operation, args...);
```

`co_dispatch_kernel` uses the same type eligibility, candidate order, decline rules,
diagnostics, and exhaustion behavior as `dispatch_kernel`.

For each eligible backend:

1. If `try_make_kernel_task` is available, the coroutine obtains a
   `KernelTaskAttempt`.
2. A clean decline continues to the next candidate, which receives any provisionally
   prepared replaceable output.
3. A completed success terminates the walk.
4. A deferred success awaits the task and then terminates the walk.
5. If no task customization exists, the coroutine calls ordinary `try_kernel`.

Operation arguments are stable lvalues owned by the calling coroutine. This avoids
copying arbitrary tensor views and keeps type probing and invocation consistent.

`co_dispatch_kernel` does not enroll epochs or await `Async<T>` operands. An async
operation wrapper must first establish the correct epoch reads and writes and await the
resolved inputs. An operation-declared replaceable output may remain unconstructed when
backend dispatch begins, allowing the attempted backend to choose its placement.

Async legality comes from epoch causality. Scheduler timing and fortunate task order
are not correctness arguments.

## Tensor, mdspan, and provider boundaries

The usual dense operation path has four layers:

```text
Tensor operation wrapper
    -> selector resolution and fixed-operand mdspec normalization
    -> operation-tag dispatch over normalized descriptors
    -> selected backend acquisition/lowering to execution-domain mdspans
    -> provider API or lower-level Uni20 module
```

### Tensor operation wrapper

The Tensor-level operation owns semantic policy that cannot be expressed by a
fixed-output leaf kernel, including:

- shape validation;
- output construction or resizing;
- aliasing rules visible at the Tensor abstraction;
- selector resolution from storage policies;
- async epoch enrollment and failure propagation.

For example, an assigning matrix product may construct or resize its output before
calling a fixed-output GEMM kernel, while an additive matrix product requires an
existing shape-compatible output.

### Fixed-descriptor dispatch boundary

A fixed-output operation-tag kernel receives normalized mdspecs and
scalar parameters. The backend may inspect their mappings, accessors, and data
descriptors before accepting an instance. Once selected, it obtains any
required domain-specific leases and resolves the operands to mdspans whose
handles and accessors are usable by its implementation.

Type probing and runtime invocation receive the same normalized descriptor
types. The dispatcher does not acquire leases. The backend that accepts the
execution domain owns acquisition and passes resolved mdspans directly to an
ordinary leaf routine rather than to another operation-tag customization.

Replaceable-output tags retain their tensor or output-storage operand until the
selected backend has supplied the missing shape and placement policy. That
backend lowers the output immediately after preparation and does not redispatch
it as a fixed-output operation tag.

The mdspan accessor defines value semantics. A pointer-shaped `data_handle_type` does
not prove that a backend may dereference the handle or pass it to a provider. A backend
may bypass indexed accessor operations only when it explicitly understands and lowers
that accessor's semantics.

### Lower-level boundary

Provider adapters and lower-level Uni20 modules receive arguments already normalized
to their contracts, commonly resolved mdspans or provider descriptors. These functions
do not participate in the `xxxx_op{}` backend walk. At this point the backend has
committed. Provider or lower-level failure is terminal and may not return a dispatch
decline.

Kernel dispatch never performs an implicit host/device transfer or an implicit dense
projection. A backend list for opaque device storage must contain only backends that can
legally access that storage.

## Fallback contract

Fallback is legal only after a clean decline.

Before returning a non-success `KernelAttempt`, a backend must not:

- write result elements or mutate a fixed or update output;
- mutate an input;
- submit provider work;
- enqueue device work;
- acquire a resource in a way visible to later candidates;
- publish a completion event;
- publish or commit a completed result;
- perform communication visible outside the attempt.

When the operation declares an output replaceable, the backend may provisionally
construct, resize, replace, or rebind that output before returning a decline. The next
candidate receives the prepared output and may reuse or replace it. If every candidate
declines, the output remains valid, but its storage, placement, shape, and element
values may reflect provisional preparation and must not be treated as a completed
result.

Preparation may inspect descriptors and create temporary local values. Any temporary
resource acquired before a possible decline must be released without externally
visible effects.

Once a backend mutates an input, a fixed or update output, or result elements; submits
work; commits a result; awaits a successful deferred task; or enters a provider
operation, failure is terminal. The dispatcher must not continue to a later backend.

Fallback also may not change the mathematical object. Symmetry metadata, block-space
metadata, local-space metadata, and leg orientation are part of the value. A
symmetry-typed operation must not silently fall through to a dense implementation.

## Diagnostics

Backends and operation tags provide static names so diagnostics can report the ordered
walk.

A failed dispatch records, for each configured backend:

- backend name;
- type acceptance (`no`, `maybe`, or `yes`);
- runtime result when the backend was attempted.

`KernelDispatchError` distinguishes:

- no type-eligible backend;
- candidates existed but all declined.

Diagnostics are observational. Enabling them must not change candidate order,
acceptance, execution, or fallback behavior.

## Backend authoring checklist

When adding a kernel to a backend:

1. Define or reuse a stable operation tag.
2. Add `kernel_accepts_types` for the exact static domain.
3. Return `yes` only when the implementation is total for that domain.
4. Provide `try_kernel` for every `maybe` or `yes` result.
5. Complete all runtime acceptance checks before any effect other than
   operation-authorized provisional preparation of a replaceable output.
6. Return a specific `KernelAttempt` for each clean-decline category.
7. Report failures after commitment through the ordinary terminal error path.
8. Add `try_make_kernel_task` only when suspension is required.
9. Keep the ordinary and coroutine implementations on the same static domain.
10. Preserve mdspan accessor semantics.
11. Do not introduce implicit transfer or symmetry-erasing fallback.
12. Add compile-time type probes and runtime success, decline, and fallback tests.

## Required tests

Focused tests should cover:

### Type eligibility

- accepted scalar, rank, accessor, and handle combinations;
- rejected output mutability;
- rejected scalar or element combinations;
- the distinction between `maybe` and `yes`;
- candidate filtering and selector order.

### Runtime attempts

- direct success;
- each meaningful decline reason;
- input and fixed/update-output preservation after decline;
- reusable provisional preparation for replaceable-output operations;
- fallback to the next candidate;
- exhaustion when all candidates decline;
- totality of every `yes` domain.

### Coroutine-aware dispatch

- ordinary blocking fallback when no task customization exists;
- clean task decline followed by fallback;
- completed success without a task;
- deferred success with suspension;
- terminal failure after task commitment;
- scheduler or execution-resource routing where applicable.

### Diagnostics

- no eligible backend;
- all candidates declined;
- ordered backend records;
- unchanged behavior with diagnostics enabled.

## Source map

The current implementation is primarily defined by:

- `src/uni20/linalg/backend_selector.hpp`
- `src/uni20/linalg/dispatch.hpp`
- `src/uni20/linalg/dispatch_error.hpp`
- `src/uni20/linalg/kernel_attempt.hpp`
- `src/uni20/linalg/async/dispatch.hpp`
- `src/uni20/linalg/async/kernel_task.hpp`

See [`backend_dispatch.md`](backend_dispatch.md) for design rationale. Distributed
composition is not part of the current contract; exploratory notes live in
[`distributed_kernel_dispatch.md`](distributed_kernel_dispatch.md).
