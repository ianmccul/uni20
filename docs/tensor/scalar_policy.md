# Scalar Type Policy

This page records the project scalar spelling and concept policy. The concrete
aliases live in `src/uni20/core/types.hpp`; scalar traits and concepts live in
`src/uni20/core/scalar_traits.hpp` and `src/uni20/core/scalar_concepts.hpp`.

## Scalar Spelling

Uni20 code should spell real and complex scalar types through the project-level
aliases when an alias exists.

The canonical real aliases are:

| alias | meaning | availability |
| --- | --- | --- |
| `uni20::float32` | `float` | always |
| `uni20::float64` | `double` | always |
| `uni20::float128` | configured binary128 real scalar | only when `UNI20_HAS_FLOAT128=1` |

`uni20::float128` is a configuration-dependent type. In the current MPLAPACK
configuration it aliases `mplapack_binary128_t`, whose concrete spelling is
selected by the installed MPLAPACK package. Code that is not gated by
`UNI20_HAS_FLOAT128` must not name `uni20::float128`.

Runtime-facing code should use `uni20::ScalarPrecision` and
`uni20::visit_scalar_precision` from `uni20/core/scalar_precision.hpp` instead
of repeating configuration guards. The precision enum always recognizes
`fp32`, `fp64`, and `fp128`; the visitor throws when a recognized precision is
not configured. The visitor is the central preprocessor boundary that maps a
runtime precision to `uni20::float32`, `uni20::float64`, or the conditional
`uni20::float128` type. `uni20::configured_scalar_precisions()` and
`uni20::configured_scalar_precision_choices()` expose the available set for
help text, diagnostics, examples, and future language bindings.

To enable this path, build MPLAPACK separately with its binary128 backend and
then point Uni20 at the resulting CMake package. Uni20 deliberately does not
download or build MPLAPACK as part of its own configure step. See
[MPLAPACK Binary128 Setup](../linalg/mplapack_binary128.md) for the exact package
build, install, Uni20 configure, and validation commands.

`uni20::complex<T>` is intentionally a type alias to `std::complex<T>`, not a
replacement class. This keeps standard-library ABI, layout expectations, and
interop behavior unchanged while giving Uni20 a single project-level spelling
for complex scalars.

The canonical complex aliases are:

| alias | meaning |
| --- | --- |
| `uni20::complex<T>` | project-level complex scalar alias |
| `uni20::complex64`, `uni20::cfloat` | `uni20::complex<float>` |
| `uni20::complex128`, `uni20::cdouble` | `uni20::complex<double>` |
| `uni20::complex256`, `uni20::cfloat128` | `uni20::complex<uni20::float128>` when `UNI20_HAS_FLOAT128=1` |

Project code, tests, examples, and documentation should use
`uni20::complex<T>` unless they are explicitly documenting or testing the alias
relationship to `std::complex<T>`, or they are at a narrow external interop
boundary that must name the standard-library type.

## Scalar Concepts

Scalar-generic code should prefer the scalar traits in `uni20/core`:

| trait/concept | use |
| --- | --- |
| `uni20::Real<T>` | real scalar constraints |
| `uni20::Complex<T>` | complex scalar constraints |
| `uni20::RealOrComplex<T>` | real-or-complex scalar constraints |
| `uni20::ScalarValued<T>` | scalar, container, or view whose recursive `value_type` resolves to a scalar |
| `uni20::RealScalarValued<T>` | scalar-valued type whose extracted scalar is real |
| `uni20::ComplexScalarValued<T>` | scalar-valued type whose extracted scalar is complex |
| `uni20::IntegerScalarValued<T>` | scalar-valued type whose extracted scalar is integer |
| `uni20::RealOrComplexScalarValued<T>` | scalar-valued type whose extracted scalar is real or complex |
| `uni20::BlasReal<T>` | real scalar with a configured dense BLAS-style backend |
| `uni20::BlasComplex<T>` | complex scalar with a configured dense BLAS-style backend |
| `uni20::LapackReal<T>` | real scalar with a configured dense LAPACK-style backend |
| `uni20::LapackComplex<T>` | complex scalar with a configured dense LAPACK-style backend |
| `uni20::LapackRealOrComplex<T>` | real or complex scalar whose underlying real precision has dense real LAPACK coverage |
| `uni20::LapackComplexReal<T>` | real precision whose `uni20::complex<T>` has dense complex LAPACK coverage |
| `uni20::make_real_t<T>` | underlying real scalar |
| `uni20::make_complex_t<T>` | complexified scalar/container type |
| `uni20::scalar_t<T>` | scalar extracted from a container-like type |
| `uni20::numeric_limits<T>` | project-level numeric limits customization point |
| `uni20::isfinite(x)` | project-level finite-value predicate for integer, real, and complex scalars |

`Scalar`, `Real`, `Complex`, `Integer`, and `RealOrComplex` constrain the type
itself. Use them for scalar-only helpers such as `uni20::conj`, `uni20::herm`,
or scalar arithmetic kernels. The `ScalarValued` family follows `scalar_t<T>`
through recursive `value_type` definitions, so it may match containers, views,
mdspans, or future tensor types. Use scalar-valued concepts when an algorithm
is generic over an object that carries scalar elements, not when the parameter
must itself be the scalar value.

`Blas*` and `Lapack*` describe configured backend support. This distinction
matters for extension scalar types: a type can be a valid Uni20 real scalar
without having BLAS or LAPACK coverage in the current build.

Reference sums and inner products use compensated accumulation in the input
scalar field. Norms use scaled sum-of-squares arithmetic in the associated real
field. Algorithms must choose their numerical accumulation method explicitly;
Uni20 does not define a universal "next wider" scalar because no such type
exists for the widest configured precision, and silent promotion would make
result and performance behavior depend on the input dtype. Integer sums are not
accepted until Uni20 defines their overflow contract.

The CPU dense matrix exponential likewise performs norm estimation and
scaling-exponent arithmetic in the matrix scalar's real field. Its overflow
protection comes from logarithmic entry bounds and prescaling before high-order
matrix powers, so binary128 support does not depend on a nonexistent wider
floating-point type.

For matrix-free algorithms, avoid duplicating scalar type information in
interfaces when it can be inferred from the vector operations. For example, the
return type of `inner_product(x, y)` names the vector scalar field, and `norm(x)`
returns the associated real scalar. Additional scalar declarations should only
be added when they encode information that cannot be inferred from the
operation interface.

Future higher-precision real types should be integrated by extending the Uni20
real scalar traits, `uni20::numeric_limits<T>`, and the required linear algebra
backends. `uni20::BlasReal<T>`, `uni20::BlasComplex<T>`,
`uni20::LapackReal<T>`, and `uni20::LapackComplex<T>` are intentionally
backend-relative and separately extensible: they mean Uni20 has configured
dense BLAS-style or LAPACK-style implementations for that scalar, not that the
type is one of the standard Fortran BLAS/LAPACK ABI scalar types. A type may
eventually satisfy only one of these concepts if Uni20 has only one backend
layer for it. For example, an MPLAPACK-enabled build may make
`mplapack_binary128_t` satisfy both `BlasReal` and `LapackReal`, while
`uni20::complex<mplapack_binary128_t>` satisfies the corresponding complex
concepts only for paths with explicit complex MPBLAS/MPLAPACK wrappers. The
BLAS side can be project overloads backed by MPBLAS entry points rather than
`s`/`d`/`c`/`z` Fortran ABI symbols.

Krylov algorithms that form dense projected Hermitian problems use
`LapackRealOrComplex`: a complex vector field can still reduce to a real
projected tridiagonal problem. Algorithms that form dense complex projected
problems use `LapackComplexReal` or `LapackComplex`, because real LAPACK support
for the underlying precision is not sufficient there.

`uni20::make_complex_t<T>` follows the Uni20 real scalar traits rather than
`std::floating_point<T>`, so extension real types can opt into the scalar model
without depending on standard-library concept recognition. If the platform
supports `std::complex<Real>` for such a real type, `uni20::complex<Real>` should
continue to be the preferred spelling. If a future backend requires a different
complex representation, that change should happen behind the project-level
alias/traits boundary rather than by scattering backend-specific complex types
through algorithms.

## Scalar Math

Scalar-generic code should use `uni20::isfinite(x)` instead of directly calling
`std::isfinite(x)` when `x` may be a Uni20 scalar. Integers are always finite,
real scalars are checked for NaN and positive/negative infinity through
`uni20::numeric_limits<T>`, and complex scalars are finite only when both
components are finite. This keeps extension scalar support behind the same
project customization points as scalar spelling and numeric limits.

Use the typed constants in `std::numbers`, such as
`std::numbers::pi_v<Real>`, when a mathematical constant participates in
scalar-generic arithmetic. Do not widen an untyped `double` constant or a
decimal `double` literal into a higher-precision scalar. The supported GNU
binary128 configuration provides full-precision typed constants for
`uni20::float128`; its tests verify that the result retains precision beyond a
widened `double`. Uni20 does not currently duplicate `std::numbers`. If a future
scalar provider cannot supply suitable typed standard constants, introduce a
project customization point when integrating that provider.

## Scalar Formatting

Scalar-generic diagnostics and presentation code should use
`uni20::format_scalar` or the corresponding presentation helpers rather than
assuming standard stream or formatter support. Trace formatting recognizes all
types satisfying Uni20's `Real` and `Complex` concepts, including
`uni20::float128` and `uni20::complex<uni20::float128>` when configured.

Trace precision is independently configurable for float32, float64, and
float128 values. The global environment variables are
`UNI20_FP_PRECISION_FLOAT32`, `UNI20_FP_PRECISION_FLOAT64`, and
`UNI20_FP_PRECISION_FLOAT128`; append `_MODULE_<MODULE>` for a module-specific
override. Complex values use the precision of their real component type.

## Numeric Limits

Scalar-generic Uni20 algorithms should use `uni20::numeric_limits<T>` rather
than naming `std::numeric_limits<T>` directly. The primary Uni20 template
inherits from `std::numeric_limits<T>`, so built-in arithmetic types use the
standard-library implementation without extra code.

For extension or library scalar types where the standard library does not
provide complete limits, specialize `uni20::numeric_limits<T>`:

```cpp
namespace uni20
{
template <> struct numeric_limits<my_real>
{
    static constexpr bool is_specialized = true;
    static constexpr int digits = /* ... */;

    static my_real epsilon();
    static my_real min();
    static my_real max();
};
} // namespace uni20
```

Do not add specializations of `std::numeric_limits` for compiler fundamental
extension types such as `__float128` or `_Float128`. Those are not
user-defined types. Keep such support behind the Uni20 customization point.
