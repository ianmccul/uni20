#pragma once

#include "scalar_traits.hpp"
#include "types.hpp"

namespace uni20
{

namespace detail
{
template <typename T>
inline constexpr bool has_builtin_blas_real_backend_v = std::same_as<T, float> || std::same_as<T, double>;

template <typename T> inline constexpr bool has_builtin_lapack_real_backend_v = has_builtin_blas_real_backend_v<T>;

template <typename T> inline constexpr bool has_mplapack_blas_real_backend_v = false;

template <typename T>
inline constexpr bool has_builtin_blas_complex_backend_v = std::same_as<T, cfloat> || std::same_as<T, cdouble>;

template <typename T> inline constexpr bool has_mplapack_blas_complex_backend_v = false;

template <typename T> inline constexpr bool has_mplapack_lapack_real_backend_v = false;

template <typename T>
inline constexpr bool has_builtin_lapack_complex_backend_v = std::same_as<T, cfloat> || std::same_as<T, cdouble>;

template <typename T> inline constexpr bool has_mplapack_lapack_complex_backend_v = false;

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
template <> inline constexpr bool has_mplapack_blas_real_backend_v<uni20::float128> = true;

template <> inline constexpr bool has_mplapack_blas_complex_backend_v<uni20::complex<uni20::float128>> = true;

template <> inline constexpr bool has_mplapack_lapack_real_backend_v<uni20::float128> = true;

template <> inline constexpr bool has_mplapack_lapack_complex_backend_v<uni20::complex<uni20::float128>> = true;
#endif

template <typename T>
inline constexpr bool has_blas_real_backend_v =
    has_builtin_blas_real_backend_v<T> || has_mplapack_blas_real_backend_v<T>;

template <typename T>
inline constexpr bool has_blas_complex_backend_v =
    has_builtin_blas_complex_backend_v<T> || has_mplapack_blas_complex_backend_v<T>;

template <typename T>
inline constexpr bool has_lapack_real_backend_v =
    has_builtin_lapack_real_backend_v<T> || has_mplapack_lapack_real_backend_v<T>;

template <typename T>
inline constexpr bool has_lapack_complex_backend_v =
    has_builtin_lapack_complex_backend_v<T> || has_mplapack_lapack_complex_backend_v<T>;
} // namespace detail

/// \brief Concept for integer scalar types (excluding char and bool).
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept Integer = is_integer_v<T>;

/// \brief Concept for real scalar types (float, double, etc.).
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept Real = is_real_v<T>;

/// \brief Concept for complex scalar types.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept Complex = is_complex_v<T>;

/// \brief Concept for numeric scalar types (integer, real, or complex).
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept Scalar = is_scalar_v<T>;

/// \brief Concept for types that are either real or complex.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept RealOrComplex = Real<T> || Complex<T>;

/// \brief Concept for BLAS-compatible real scalar types.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept BlasReal = Real<T> && detail::has_blas_real_backend_v<T>;

/// \brief Concept for BLAS-compatible complex scalar types.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept BlasComplex = Complex<T> && detail::has_blas_complex_backend_v<T>;

/// \brief Concept for all BLAS-compatible scalar types.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept BlasScalar = BlasReal<T> || BlasComplex<T>;

/// \brief Concept for real scalar types with a configured dense LAPACK backend.
/// \details This names the LAPACK requirement explicitly for dense
///          factorizations, eigensolvers, Schur decompositions, and related
///          projected Krylov subspace kernels. It is intentionally separate
///          from `BlasReal` so dense BLAS and dense LAPACK backend coverage can
///          diverge for extension scalar types.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept LapackReal = Real<T> && detail::has_lapack_real_backend_v<T>;

/// \brief Concept for complex scalar types with a configured dense LAPACK backend.
/// \details This is intentionally separate from `BlasComplex` so dense BLAS and
///          dense LAPACK backend coverage can diverge for extension scalar
///          types.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept LapackComplex = Complex<T> && detail::has_lapack_complex_backend_v<T>;

/// \brief Concept for real or complex scalar fields with real LAPACK coverage.
/// \details This is useful for algorithms such as Hermitian Lanczos where the
///          application-space vectors may be complex, but the projected dense
///          problem is real and only requires LAPACK support for the underlying
///          real precision.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept LapackRealOrComplex = LapackReal<T> || (Complex<T> && LapackReal<make_real_t<T>>);

/// \brief Concept for real precisions whose Uni20 complex scalar has LAPACK coverage.
/// \details This constrains algorithms that template on the underlying real
///          precision while forming dense complex projected problems.
/// \tparam T Real precision to test.
/// \ingroup core_math
template <typename T>
concept LapackComplexReal = Real<T> && LapackComplex<uni20::complex<T>>;

/// \brief Concept for scalar types with a configured dense LAPACK backend.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept LapackScalar = LapackReal<T> || LapackComplex<T>;

/// \brief Concept for a type that is a scalar, or has a scalar value_type.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept HasScalar = has_scalar_v<T>;

/// \brief Concept for types whose `scalar_t<T>` is an integer.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept HasIntegerScalar = has_integer_scalar_v<T>;

/// \brief Concept for types whose `scalar_t<T>` is real.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept HasRealScalar = has_real_scalar_v<T>;

/// \brief Concept for types whose `scalar_t<T>` is complex.
/// \tparam T Type to test.
/// \ingroup core_math
template <typename T>
concept HasComplexScalar = has_complex_scalar_v<T>;

} // namespace uni20
