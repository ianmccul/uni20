#pragma once

#include <uni20/config.hpp>
#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
#include <mplapack_config.h>
#endif
#include <complex>
#include <concepts>

namespace uni20
{

// Fundamental types used throughout the library

/// \brief Signed size and index type, used for tensor extents and indexing.
/// \note Using signed values avoids unnecessary conversions in loop logic.
using size_type = std::ptrdiff_t;
using index_type = std::ptrdiff_t;

// Type aliases for explicit precision and complex values
using float32 = float;
using float64 = double;

/// \brief Project-level complex scalar spelling.
/// \details This is intentionally an alias to `std::complex`, not a wrapper. Code in Uni20 should spell complex
///          scalar types as `uni20::complex<T>` so future scalar-policy changes have one namespace-level hook while
///          preserving standard-library ABI and interop today.
/// \tparam Real Underlying real scalar type.
/// \ingroup core_math
template <typename Real> using complex = std::complex<Real>;

using complex64 = complex<float>;
using complex128 = complex<double>;

using cfloat = complex<float>;
using cdouble = complex<double>;

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
/// \brief Configured project-level binary128 real scalar.
/// \details This alias is available only when Uni20 is configured with a
///          binary128 provider. In the MPLAPACK configuration it names
///          `mplapack_binary128_t`, whose concrete C++ spelling is selected by
///          the installed MPLAPACK package.
/// \ingroup core_math
using float128 = mplapack_binary128_t;

/// \brief Configured project-level binary128 complex scalar.
/// \ingroup core_math
using complex256 = complex<float128>;

using cfloat128 = complex256;
#endif

/// \brief Trait for extracting the element type from a proxy reference.
///
/// This is a customization point: user-defined proxy types should specialize this template
/// for their proxy wrapper `Proxy<T>`, mapping it to `T`.
///
/// The default implementation removes only the reference qualifier (not CV).
/// CV and reference qualifications are handled automatically by the wrapper logic.
template <typename R> struct remove_proxy_reference : std::remove_reference<R>
{};

/// \brief Detects whether a type is considered a proxy reference.
///
/// Evaluates to true if `remove_proxy_reference` changes the type,
/// i.e., if the transformation produces a different type than the CV-ref stripped input.
template <typename R>
constexpr bool is_proxy_reference_v =
    !std::is_same_v<typename remove_proxy_reference<std::remove_cvref_t<R>>::type, std::remove_cvref_t<R>>;

/// \brief Extracts the underlying value type of a proxy (or normal) reference.
///
/// If the type is detected as a proxy reference via `is_proxy_reference_v`, removes the proxy
/// wrapper and CV qualifiers. Otherwise, removes only the reference qualifier.
template <typename R>
using remove_proxy_reference_t =
    std::conditional_t<is_proxy_reference_v<R>, typename remove_proxy_reference<std::remove_cvref_t<R>>::type,
                       std::remove_reference_t<R>>;

} // namespace uni20
