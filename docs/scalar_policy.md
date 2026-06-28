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

`Real`, `Complex`, and `RealOrComplex` describe scalar categories. `Blas*` and
`Lapack*` describe configured backend support. This distinction matters for
extension scalar types: a type can be a valid Uni20 real scalar without having
BLAS or LAPACK coverage in the current build.

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
