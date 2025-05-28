#pragma once

#include "config.hpp"
#include <complex>
#include <concepts>

namespace uni20
{

// The default size_type and index_type.  Perfer having size_type signed as well, so that
// we can use ordinary integers as loop variables without unwanted conversions
using size_type = std::ptrdiff_t;
using index_type = std::ptrdiff_t;

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

// aliases for types

using float32 = float;
using float64 = double;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

#if defined(HAVE_FLOAT128)
using float128 = __float128;
#endif

// is_real
// A trait for determining whether a type is treated as floating point
template <typename T> struct is_real_t : std::false_type
{};
template <> struct is_real_t<float> : std::true_type
{};
template <> struct is_real_t<double> : std::true_type
{};
template <> struct is_real_t<long double> : std::true_type
{};

template <typename T> constexpr bool is_real = is_real_t<std::remove_cvref_t<T>>::value;

// is_complex
// A trait for determining whether a type is treated as complex,
// i.e. std::complex, or some similar extension
template <typename T> struct is_complex_t : std::false_type
{};
template <typename T> struct is_complex_t<std::complex<T>> : std::true_type
{};

template <typename T> constexpr bool is_complex = is_complex_t<std::remove_cvref_t<T>>::value;

// is_integer
// A trait to check for an integer type (excluding bool)
template <typename T> struct is_integer_t : std::bool_constant<std::integral<T> && (!std::same_as<T, bool>)>
{};

template <typename T> constexpr bool is_integer = is_integer_t<std::remove_cvref_t<T>>::value;

//
// Concepts
//
// Integer        - integral type (same as std::integral, but could be extended by user custom types)
// Real           - a real floating point number (including extensions that are floating-point-like)
// Complex        - complex floating point; either std::complex<RealType> or some complex-like extension
// RealOrComplex  - either real or complex
// Numeric        - either real or complex or integer
// BlasReal       - real types accepted by standard BLAS (i.e. single and double precision)
// BlasComplex    - complex types accepted by standard BLAS (i.e. single and double precision complex)
// BlasScalar     - union of BlasReal and BlasComplex
//

template <typename T>
concept Integer = is_integer<T>;

template <typename T>
concept Real = is_real<T>;

template <typename T>
concept Complex = is_complex<T>;

template <typename T>
concept RealOrComplex = Real<T> || Complex<T>;

template <typename T>
concept Numeric = Real<T> || Complex<T> || Integer<T>;

template <typename T>
concept BlasReal = std::same_as<T, float> || std::same_as<T, double>;

template <typename T>
concept BlasComplex = std::same_as<T, cfloat> || std::same_as<T, cdouble>;

// Define a concept for BLAS scalar types (either a BLAS real or a BLAS complex type).
template <typename T>
concept BlasScalar = BlasReal<T> || BlasComplex<T>;

// make_real
// metafunction to get a real type corresponding to some scalar (real or complex)
// Customize this for extension types
template <typename T> struct make_real;

template <Real T> struct make_real<T>
{
    using type = T;
};

template <typename T> struct make_real<std::complex<T>>
{
    using type = T;
};

template <typename T> using make_real_type = make_real<T>::type;

// make_complex
// metafunction to get a complex type corresponding to some scalar (real or complex).
// If the type T is already complex then return T as-is.
// Customize this for extension types.
template <typename T> struct make_complex;

template <typename T>
  requires std::floating_point<T>
struct make_complex<T>
{
    using type = std::complex<T>;
};

template <typename T>
  requires Complex<T>
struct make_complex<T>
{
    using type = T;
};

template <typename T> using make_complex_type = make_complex<T>::type;

// numeric_type
// Recursively extracts the underlying numeric type from a container.
// For instance, for std::vector<std::vector<int>>, it returns int.

template <typename T> struct numeric_type;

template <typename T>
  requires Numeric<T>
struct numeric_type<T>
{
    using type = T;
};

// We want to avoid string-like objects from having a numeric_type. Naively std::string
// does, because it is a container and the value_type is char, which is an integral type.
// But we want to consider strings as a basic (non-scalar, non-numeric) type.
namespace detail
{
// A trait to detect if T is an instantiation of std::basic_string
template <typename T> struct is_std_basic_string : std::false_type
{};

template <typename CharT, typename Traits, typename Alloc>
struct is_std_basic_string<std::basic_string<CharT, Traits, Alloc>> : std::true_type
{};

template <typename T> constexpr bool is_std_basic_string_v = is_std_basic_string<T>::value;

// Now define the concept is_string:
template <typename T>
concept is_string = is_std_basic_string_v<T>;

} // namespace detail

template <typename T>
  requires requires { typename T::value_type; } &&
           (!Numeric<T>) && Numeric<typename numeric_type<typename T::value_type>::type> && (!detail::is_string<T>)
struct numeric_type<T> : numeric_type<typename T::value_type>
{};

template <typename T> using numeric_t = typename numeric_type<T>::type;

template <typename T>
concept HasNumericType = requires { typename numeric_type<T>::type; };

// scalar_type
// Recursively extracts the underlying scalar type from a container.
// For instance, for std::vector<std::vector<double>>, it returns double.

template <typename T> struct scalar_type;

template <typename T>
  requires RealOrComplex<T>
struct scalar_type<T>
{
    using type = T;
};

template <typename T>
  requires requires { typename T::value_type; } &&
           (!Complex<T>) && RealOrComplex<typename scalar_type<typename T::value_type>::type>
struct scalar_type<T> : scalar_type<typename T::value_type>
{};

template <typename T> using scalar_t = typename scalar_type<T>::type;

template <typename T>
concept HasScalarType = requires { typename scalar_type<T>::type; };

} // namespace uni20
