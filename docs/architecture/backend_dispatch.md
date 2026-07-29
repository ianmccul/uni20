# Backend dispatch rationale

This document records the rationale behind Uni20's kernel-dispatch design. It is
non-normative. The current contract is defined by
[`kernel_dispatch.md`](kernel_dispatch.md) and the corresponding source under
`src/uni20/linalg/`.

## Why a generic kernel dispatcher exists

Dense linear-algebra operations have several possible implementations:

- generic indexed reference kernels;
- host BLAS and LAPACK providers;
- device providers such as cuBLAS;
- specialized implementations for particular layouts, accessors, or scalar types.

The Tensor front end should not contain a separate hand-written backend walk for every
operation. Conversely, provider wrappers should not decide Tensor-level shape policy,
storage transfers, or async epoch dependencies.

The dispatcher provides the narrow boundary between those layers:

```text
Tensor operation policy
    -> ordered kernel dispatch
    -> provider or reference leaf
```

## Why operation tags are values

Earlier designs can be tempted toward one dispatcher type or one virtual interface per
operation. Operation tags avoid that proliferation.

A backend implements overloads for a lightweight tag:

```cpp
kernel_accepts_types(backend, gemm_op{}, args...)
try_kernel(backend, gemm_op{}, args...)
```

This has several advantages:

- one generic walk serves every kernel family;
- a backend can implement any subset of operations;
- unsupported operations disappear through overload resolution;
- operation-specific state may be carried by the tag when needed;
- diagnostics have a stable operation identity.

The operation tag is not the user-facing operation. Tensor-level wrappers still own
shape changes, output preparation, and semantic validation.

## Why selectors contain ordered backend values

Backend order is policy. A selector such as

```cpp
backend_list{
    LapackBackend{},
    BlasBackend{},
    CpuReferenceBackend{},
};
```

means "try these implementations in this order."

Selectors contain values rather than only backend types because backend configuration
and shared state may be meaningful. Normalization preserves those values and their
order.

Selectors are resolved from Tensor storage policies where possible. This prevents
backend selection from becoming a runtime examination of Tensor values and keeps
storage-domain compatibility explicit.

## Why eligibility is split into type and value stages

A single runtime `try_kernel` interface could theoretically reject everything. That
would force the compiler to instantiate implementations for argument types they cannot
possibly support and would make ordinary C++ errors late and noisy.

Uni20 therefore separates:

1. **Type eligibility**, through `kernel_accepts_types`.
2. **Runtime acceptance**, through `try_kernel`.

The type stage removes impossible candidates before their implementations are
instantiated. The runtime stage handles properties such as shape, strides, aliasing,
device identity, and provider availability.

This split also supports useful compile-time tools:

- aggregate probing;
- candidate-list filtering;
- conformance tests for backend coverage.

## Why the result is `no`, `maybe`, or `yes`

A Boolean type result cannot distinguish a backend that might decline from one that is
total for the admitted types.

The three states encode that distinction:

- `no`: never a candidate for these types;
- `maybe`: runtime values determine support;
- `yes`: every valid value in the admitted type domain is supported.

`yes` is useful for reference backends and other complete implementations. It provides a
checked endpoint for an ordered walk: if a `yes` backend declines, the implementation
has violated its declared contract.

The dispatcher does not reinterpret "last backend" as `yes`. Totality is a property of
the implementation, not its position in a selector.

## Why `kernel_accepts_types` may probe a constrained implementation

Backend type support is sometimes naturally defined by a lower-level constrained
attempt function. In that case, testing whether the function is callable avoids
duplicating a long type computation:

```cpp
requires {
    {
        detail::try_gemm(output, alpha, lhs, rhs, beta)
    } -> std::same_as<KernelAttempt>;
}
```

This is sound when the probed declaration contains the real static constraints. It is
not sound to probe an unconstrained function template and expect the expressions in its
body to participate in callability.

The design goal is one source of truth, not one mandatory spelling:

- probe a constrained implementation when it already defines the static domain;
- use a named concept or direct requirements when that is clearer;
- do not maintain equivalent independent predicates without a reason.

## Why runtime declines are values, not exceptions

A runtime decline is an expected part of backend selection. Examples include an
unsupported stride pattern or a temporarily unavailable optional provider.

`KernelAttempt` gives these outcomes a small allocation-free vocabulary. Exceptions and
ordinary error channels are reserved for terminal failures.

This distinction matters because fallback is only correct before execution has
started. A returned decline therefore guarantees that inputs and fixed or update
outputs are unchanged and no work or completed result was committed. An operation may
explicitly permit provisional construction, resizing, or replacement of a replaceable
output; later candidates receive that prepared output.

## Why clean decline is a hard invariant

A fallback walk presents the same logical operation to later candidates. Inputs and
existing values that participate in the operation must therefore remain untouched.
Storage and shape of an overwrite-only replaceable output are not participating values
and may be provisionally prepared for the next candidate.

The forbidden sequence is:

```text
backend A mutates an input, participating output value, or submits work
backend A reports decline
backend B receives the partially executed operation
```

Once result elements or participating values are mutated, work is submitted, or a
completed result is committed, backend A owns the operation. Any later failure is
terminal. Preparing only the shape or storage of an operation-declared replaceable
output does not commit its result.

This rule applies equally to host providers, device providers, and future communication
layers.

## Why dispatch uses stable lvalues

The dispatcher probes and invokes backend customizations with stable lvalue arguments.

This avoids repeatedly forwarding or moving an operand during an ordered walk. It also
lets later candidates receive the same stable argument objects after an earlier clean
decline. An operation-declared replaceable output may have a newly prepared descriptor
inside that stable object.

The type probe deliberately mirrors this invocation shape. It removes a reference and
then forms an lvalue reference while preserving cv-qualification.

## Why Tensor policy is explicit at the dispatch boundary

Kernel dispatch is tensor-view-oriented, while operation tags declare whether an
output is fixed, updated, or replaceable. A Tensor operation may need to:

- construct or resize an output;
- choose resources for that output;
- establish aliasing rules at the Tensor abstraction;
- enroll async reads and writes;
- preserve symmetry or block structure.

Those permissions are operation semantics, while exact placement requirements may be
backend-specific. The Tensor wrapper establishes the operand roles and passes
`TensorView`, `DeviceTensorView`, or potentially unconstructed replaceable-output
storage into the dispatcher. A backend may provisionally prepare a replaceable output,
then acquire and lower the resulting views to execution-domain mdspans. This keeps the
same backend implementation usable from synchronous and coroutine-aware front ends.

## Why mdspan accessors are part of eligibility

An mdspan's accessor defines the values observed through the view. Its data handle alone
does not prove that a provider can read or write the underlying storage.

A backend may accept:

- the default accessor;
- a custom accessor it explicitly lowers;
- a semantic accessor represented by a provider transform.

Otherwise it must decline or leave the types ineligible. A generic indexed backend may
remain eligible when its expressions respect the accessor.

This prevents a fast provider path from silently changing the mathematical operation.

## Why storage transfer is not fallback

Backend fallback chooses among implementations that can legally operate on the same
resolved arguments. It does not change where the arguments live.

A host backend cannot serve as an implicit fallback for opaque device storage. A
device-to-host transfer is an explicit operation with allocation, synchronization, and
cost semantics. It must be represented by the operation or an explicit adapter, not
hidden inside a decline path.

The same reasoning forbids an implicit dense projection for symmetry-typed values.

## Why coroutine support is an optional backend customization

Many kernels can run directly when a coroutine reaches them. Requiring every backend to
wrap ordinary work in a coroutine would add boilerplate without improving semantics.

Some backends do need suspension, especially for bounded execution-resource admission.
For those cases, `try_make_kernel_task` returns either:

- a clean decline;
- completed success;
- a deferred task.

The ordinary `try_kernel` remains required for the accepted static domain. This keeps
the core backend contract available to synchronous dispatch and gives coroutine-aware
dispatch a blocking fallback.

## Why task commitment stops fallback

A successful `KernelTaskAttempt` with a deferred task means the backend has accepted the
operation. Awaiting that task may acquire resources, submit work, and mutate output.

Failures from the task are therefore terminal. They are not converted back into
`KernelAttempt`, because doing so would permit fallback after commitment.

## Why dynamic dispatch is a separate boundary

Ordinary templated C++ callers benefit from a compile-time error when no backend accepts
their argument types.

Runtime-erased interfaces cannot always preserve that constraint. Python bindings and
similar boundaries must remain callable and report the missing type support at runtime.

`dynamic_dispatch_kernel` exists for that case. Keeping it separate prevents the
runtime-erased requirement from weakening ordinary C++ diagnostics.

## Alternatives rejected

### One virtual backend interface

A virtual interface would require a closed operation and argument type model or a large
type-erasure layer. Uni20's scalar, mdspan, accessor, and operation types are better
served by constrained overload resolution.

### One dispatcher per operation

Per-operation walks duplicate ordering, fallback, diagnostics, and async behavior. They
also drift as new backend rules are added.

### Runtime backend enumeration only

An enum loses backend-specific state and still requires a secondary mechanism for
operation overloads and static type support.

### Exceptions for ordinary decline

Expected layout or instance rejection would become expensive and would blur the
critical boundary between clean fallback and terminal failure.

### Implicit transfer or materialization during fallback

This hides allocation, synchronization, and semantic conversion. Such work must be
explicit in the operation contract.

## Maintenance rule

The dispatch mechanism is considered settled. Changes to customization-point shapes,
acceptance semantics, fallback, selector resolution, coroutine commitment, or
diagnostics must update [`kernel_dispatch.md`](kernel_dispatch.md), focused tests, and
the relevant source documentation in the same change.

Historical rationale may be refined here, but this file must not become a second
normative specification or a status inventory.
