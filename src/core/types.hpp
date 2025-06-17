#pragma once

#include "config.hpp"
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

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

#if defined(HAVE_FLOAT128)
using float128 = __float128;
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
constexpr bool is_proxy_v =
    !std::is_same_v<typename remove_proxy_reference<std::remove_cvref_t<R>>::type, std::remove_cvref_t<R>>;

/// \brief Extracts the underlying value type of a proxy (or normal) reference.
///
/// If the type is detected as a proxy via `is_proxy_v`, removes the proxy
/// wrapper and CV qualifiers. Otherwise, removes only the reference qualifier.
template <typename R>
using remove_proxy_reference_t =
    std::conditional_t<is_proxy_v<R>, typename remove_proxy_reference<std::remove_cvref_t<R>>::type,
                       std::remove_reference_t<R>>;

} // namespace uni20
