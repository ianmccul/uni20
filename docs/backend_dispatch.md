# Backend Dispatch Design

This note records the preferred direction for future Uni20 backend work. It is
planning guidance, not a statement that every current backend already follows
this pattern.

## Core Rule

Keep vendor capability out of the public algorithm contract.

Public operations should describe Uni20 semantics: copy, scale, multiply,
contract, solve, factorize, exponentiate, transform, or reduce. Backend-specific
facts such as BLAS scalar support, CUDA residency, cuSOLVER workspace rules,
integer width, vendor handles, streams, and ABI details should stay in backend
adapters.

## Dispatch Shape

Use a three-stage dispatch path for optimized backends:

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

## Compile-Time Capability

`maybe_can_*` checks only type-level facts. It should not inspect runtime values.

Examples:

- scalar category or exact scalar type
- rank and static extent requirements
- layout policy
- accessor policy and mutability
- memory-space or storage policy
- whether a backend call is well-formed for these C++ types

Prefer traits or `constexpr` functions whose template arguments match the
operation being lowered:

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

## Runtime Attempt

`try_*` checks runtime facts and performs the backend call if all checks pass.

Examples:

- extents fit the backend index type
- strides and increments can be represented by the backend API
- layout can be represented as an order or transpose flag
- operands do not alias in an unsupported way
- memory is resident on the required device
- the resolved vendor library version supports the active device architecture
  and requested operation
- handles, streams, workspaces, and scratch buffers are available
- backend-specific status codes indicate success

If the runtime checks pass, `try_*` performs the backend call and reports success.
If any check fails, it reports failure without changing the semantic contract of
the public operation.

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
- Put backend eligibility in `maybe_can_*`, not in public-facing overloads.
- Put runtime validation and the actual backend call in `try_*`.
- Keep vendor ABI details in adapters.
- Add tests that cover backend success, forced generic execution for
  backend-eligible inputs, and fallback behavior when practical.
- Compare backend results with the generic implementation for representative
  inputs.
- Document when a backend path is an optimization only and not a semantic
  requirement.
