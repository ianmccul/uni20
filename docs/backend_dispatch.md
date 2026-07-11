# Backend Dispatch Design

This note records the preferred direction for future Uni20 backend work. It is
planning guidance, not a statement that every current backend already follows
this pattern. The three-stage pattern described here is generalized to an
ordered, overridable backend list — with operation tags, detected customization
points, and nesting for distributed execution — in `kernel_dispatch.md`; this
note remains the reference for the per-backend compile-time capability /
runtime-attempt / fallback contract.

The static capability CPO name used here is `kernel_accepts_types(...)`. It is
inspired by the Kokkos/stdBLAS `maybe_can_*` pattern, but names the Uni20 role
more directly: the backend accepts these C++ argument types. Runtime view
instances are still checked by `try_kernel(...)`.

## Core Rule

Keep vendor capability out of the public algorithm contract.

Public operations should describe Uni20 semantics: copy, scale, multiply,
contract, solve, factorize, exponentiate, transform, or reduce. Backend-specific
facts such as BLAS scalar support, CUDA residency, cuSOLVER workspace rules,
integer width, vendor handles, streams, and ABI details should stay in backend
adapters.

## Dispatch Shape

Use a three-stage dispatch path for optimized backends. Conceptually:

```cpp
template <typename... Args>
auto algorithm(Args&&... args)
{
  if constexpr (maybe_can_backend_algorithm_v<std::remove_cvref_t<Args>...>)
  {
    if (try_backend_algorithm(std::forward<Args>(args)...))
    {
      return;
    }
  }

  generic_algorithm(std::forward<Args>(args)...);
}
```

The same structure also applies to functions that return a value. In that case,
`try_backend_algorithm` should return an object such as `std::optional<Result>`,
or should write into an output argument and return `bool`.

In the full kernel-dispatch design, this is expressed once for every operation
through an operation tag and customization points:

```cpp
struct gemm_op {};

template <class Backend, class Op, class... Args>
consteval KernelTypeAcceptance kernel_type_acceptance();
// detects optional kernel_accepts_types(...) and constrained try_kernel(...)

template <class Backend, class Op, class... Args>
bool try_kernel(Backend, Op, Args&&...); // implemented only for supported operations
```

A backend that does not provide a usable `try_kernel(Backend, Op, ...)`
overload does not implement that operation and is skipped. A backend may provide
`consteval KernelTypeAcceptance kernel_accepts_types(Backend const&, Op const&,
Args&...)` for extra type-level eligibility. The CPO is written as an ordinary
compile-time function, so it does not need a tuple or explicit type-pack
argument. This keeps the backend walk generic and avoids a separate
`dispatch_gemm`, `dispatch_scale`, `dispatch_svd`, etc. implementation for
every kernel.

`KernelTypeAcceptance` is a tri-state:

```cpp
enum class KernelTypeAcceptance {
  no,     // invalid for these types; do not form the try_kernel expression
  maybe, // types are plausible, but runtime values decide
  yes    // this backend should always accept these types; try_kernel may assert
};
```

Callers may select either an ordered backend-list value or a single backend
value. The single backend form is normalized to a one-element list, so
forced-backend tests and benchmarks do not need to spell a one-entry list.
Spelling such as `BackendChain<LapackBackend, CpuBackend>` is equivalent in
spirit to an ordered backend-list selector: try LAPACK first, then the CPU
fallback. It should not introduce a separate state parameter or a second
dispatch protocol.

The list has a static type order, but it is not only a type list. Default
selectors should normally contain stateless backend tags. Explicit overrides
may carry operation context that is not intrinsic to an operand:

```cpp
struct BlasBackend {};
struct CublasBackend {
  cudaStream_t stream;
  math_mode_t math_mode;
};
```

Memory kind and runtime placement belong to the operand's accessor-defined data
handle, not its default selector. CUDA streams, MPI communicators, workspace
policy, algorithm options, or multiprecision settings may justify stateful
explicit selectors. That later design should still keep the CPO call shape
backend-first.

If several backend-list entries share operation context, the selector can store
that state once and hand an appropriate backend view/value to the CPO. This is a
selector implementation detail, not part of the leaf-kernel CPO signature.

For a backend chain, the conservative type acceptance rule is:

- if every candidate is `no`, the chain is `no`;
- if the first candidate that can run is `yes`, the chain may report `yes`;
- otherwise the chain should report `maybe`, because runtime checks are needed
  somewhere inside the chain.

If a future selector needs type-level state composition, it may still use a
helper like:

```cpp
template <class BackendList>
struct backend_selector_state;
```

but that should not force every backend CPO to receive a separate `State&`
parameter.

## Compile-Time Capability

`kernel_accepts_types(...)` checks only type-level facts. It should not inspect
runtime values. The generic dispatcher may call it with private type-probe
lvalues that have no real runtime object behind them.

The tri-state matters:

- `no`: this backend is not even a candidate for these C++ types. The dispatcher
  must skip it without requiring `try_kernel(...)` to be well-formed.
- `maybe`: this backend can compile for these types, but runtime facts such as
  strides, aliasing, device placement, library availability, or matrix shape
  still decide whether this instance can run.
- `yes`: this backend is structurally guaranteed for these types. The dispatcher
  can treat a `false` return from `try_kernel(...)` as a backend bug, and assert
  in debug builds.

Examples:

- scalar category or exact scalar type
- rank and static extent requirements
- layout policy
- accessor policy and mutability
- memory-space or storage policy
- whether a backend call is well-formed for these C++ types

Prefer traits or `constexpr` functions whose template arguments match the
operation being lowered. In the operation-tag model, those facts can live either
in a constrained `try_kernel` overload or in an optional
`kernel_accepts_types(...)` customization:

```cpp
template <typename Scalar, typename Mdspan>
inline constexpr bool maybe_can_blas_scale_v =
    blas_scalar<std::remove_cv_t<typename Mdspan::value_type>> &&
    std::convertible_to<Scalar, typename Mdspan::value_type> &&
    blas_layout<typename Mdspan::layout_type> &&
    blas_accessor<typename Mdspan::accessor_type>;
```

The purpose of this layer is to keep invalid backend code out of overload
resolution and out of compiled control flow.

Backend `try_kernel` overloads must be genuinely constrained. Do not rely on
invalid statements inside the function body to reject unsupported types; the
generic dispatcher can only skip a backend if overload resolution or a
`kernel_accepts_types(...)` check makes the operation unavailable.

## Runtime Attempt

`try_kernel(...)` / `try_*` checks runtime facts and performs the backend call if
all checks pass.

Examples:

- extents fit the backend index type
- strides and increments can be represented by the backend API
- layout can be represented as an order or transpose flag
- operands do not alias in an unsupported way
- memory is resident on the required device
- the backend value or selector context is compatible with the operands, for
  example CUDA device, stream/scheduler target, or workspace allocator
- the resolved vendor library version supports the active device architecture
  and requested operation
- handles, streams, workspaces, and scratch buffers are available
- backend-specific status codes indicate success

If the runtime checks pass, `try_kernel(...)` performs the backend call and
reports success. If any check fails, it reports failure without changing the
semantic contract of the public operation.

Backend state has two broad categories:

- **Operand state** describes storage semantics and placement: memory domain,
  CUDA device, and allocator/resource. It belongs to the tensor storage and its
  accessor-defined data handle.
- **Operation context** affects execution rather than operand identity: a stream,
  communicator, workspace ownership, precision setting, or forced algorithm.
  It may be carried by an explicit selector override.
- **Advisory state** guides optimization only: preferred algorithm, tile shape,
  math mode, fusion hint, workspace budget, or "try this path first".

Operand compatibility and operation context should be checked before side
effects and before scratch allocation where practical. Advisory state may vary
across backend-list entries and may be ignored by backends that do not
understand it.

## Fallback Semantics

Every optimized path needs a semantic fallback. The fallback is the reference
behavior for correctness and documentation. It may be a generic CPU loop, an
existing Uni20 kernel, or another already-supported backend.

The fallback should not be treated as an error path. It is the expected path for
valid inputs that a specific backend cannot represent.

Vendor-library availability is also a runtime capability, not only a build
configuration fact.  For CUDA libraries this includes the resolved shared
library version, CUDA ABI, active device architecture, and operation-specific
support.  See `cuda_backend_libraries.md` for the versioned CUDA library model.

## Testing Against Fallbacks

A backend path that is always available for a type family can hide bugs in the
generic implementation. For example, if `try_backend_algorithm` always succeeds
for `double` contiguous matrices, normal tests for those inputs will never
execute `generic_algorithm`.

Unit tests must be able to exercise the generic implementation directly,
including for inputs where an optimized backend is expected to run. Do not test
the generic path only on backend-ineligible inputs. Otherwise, the generic
implementation may remain unexecuted for common scalar, layout, and stride
combinations until a future build, platform, dependency version, or runtime
condition disables the backend path.

Use the generic path as the correctness oracle for backend-specific tests, and
test backend results against the generic calculation for representative shapes,
strides, scalar types, and aliasing cases.

When practical, include tests that force or select the fallback path even for
inputs that an optimized backend would normally handle. This keeps the fallback
maintained as real code rather than as an untested emergency path.

## Backend Adapters

Foreign APIs should be wrapped behind Uni20 adapter functions before algorithm
code calls them.

Examples of adapter responsibilities:

- vendor headers and feature macros
- C, Fortran, C++, or CUDA ABI details
- symbol naming and name mangling
- index-width conversion and range checks
- handle, stream, queue, and workspace management
- status-code conversion to Uni20 errors
- vendor-specific layout, transpose, and leading-dimension conventions

For BLAS-like libraries, prefer a C API when one is available. If only a Fortran
ABI is available, put the Fortran wrapper behind a Uni20 C-like adapter so
algorithm lowering code does not depend on symbol mangling or Fortran integer
details.

The low-level LAPACK facade currently follows a two-layer convention:

- `uni20::lapack::unchecked` contains type-safe, provider-backed overloads that
  return the raw LAPACK `INFO` value. These wrappers are marked `[[nodiscard]]`
  and are the right layer for algorithms that want to handle singularity,
  non-convergence, workspace queries, condition warnings, or fallback policy
  themselves.
- `uni20::lapack` contains thin checked wrappers that call the corresponding
  unchecked overload and throw with diagnostic context for ordinary fatal
  statuses such as invalid arguments, singular factors, or convergence failure.
  This layer should not contain algorithmic fallback policy. Higher-level
  helpers such as `svd`, `solve`, or tensor truncation routines can call
  `unchecked` directly when they need trial algorithms or richer recovery.

The provider is a physical implementation detail, not part of the public call
surface. Standard LAPACK overloads and MPLAPACK binary128 overloads should
present the same `uni20::lapack::unchecked::<routine>` spelling when they
provide the same operation. Future global kernel dispatch can then detect
compile-time callability and runtime suitability without hard-coding provider
names into algorithm code.

## Applicability

This pattern is not BLAS-specific. Use it for future optimized paths involving:

- BLAS, LAPACK, cuBLAS, cuSOLVER, and similar dense linear algebra libraries
- FFT, sparse, eigensolver, and tensor-network-specific kernels
- TBB or other CPU parallel kernels
- CUDA, HIP, SYCL, or other device backends
- expression lowering and fusion

## Development Guidance

- Keep the public overload generic unless the operation semantics themselves
  require a constraint.
- Put backend eligibility in constrained `try_kernel` overloads or optional
  `kernel_accepts_types(...)` checks, not in public-facing overloads.
- Put runtime validation and the actual backend call in `try_kernel(...)`.
- Keep vendor ABI details in adapters.
- Add tests that cover backend success, forced generic execution for
  backend-eligible inputs, and fallback behavior when practical.
- Compare backend results with the generic implementation for representative
  inputs.
- Document when a backend path is an optimization only and not a semantic
  requirement.
