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

// We sometimes need to extract the element_type from a proxy reference
template <typename R> struct remove_proxy_reference
{
    using type = std::remove_cv_t<std::remove_reference_t<R>>;
};

template <typename R> using remove_proxy_reference_t = typename remove_proxy_reference<R>::type;

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

template <typename T> constexpr bool is_real = is_real_t<T>::value;

// is_complex
// A trait for determining whether a type is treated as complex,
// i.e. std::complex, or some similar extension
template <typename T> struct is_complex_t : std::false_type
{};
template <typename T> struct is_complex_t<std::complex<T>> : std::true_type
{};

template <typename T> constexpr bool is_complex = is_complex_t<T>::value;

// is_integer
// A trait to check for an integer type (excluding bool)
template <typename T> struct is_integer_t : std::bool_constant<std::integral<T> && (!std::same_as<T, bool>)>
{};

template <typename T> constexpr bool is_integer = is_integer_t<T>::value;

//
// Concepts
//
// integer        - integral type (same as std::integral)
// real           - a real floating point number (including extensions that are floating-point-like)
// complex        - complex floating point; either std::complex<RealType> or some complex-like extension
// scalar         - either real or complex
// numeric        - either real or complex or integer
// blas_real      - real types accepted by standard BLAS (i.e. single and double precision)
// blas_complex   - complex types accepted by standard BLAS (i.e. single and double precision complex)
// blas_scalar    - union of blas_real and blas_complex
//

template <typename T>
concept integer = is_integer<T>;

template <typename T>
concept real = is_real<T>;

template <typename T>
concept complex = is_complex<T>;

template <typename T>
concept scalar = real<T> || complex<T>;

template <typename T>
concept numeric = real<T> || complex<T> || integer<T>;

template <typename T>
concept blas_real = std::same_as<T, float> || std::same_as<T, double>;

template <typename T>
concept blas_complex = std::same_as<T, cfloat> || std::same_as<T, cdouble>;

// Define a concept for BLAS scalar types (either a BLAS real or a BLAS complex type).
template <typename T>
concept blas_scalar = blas_real<T> || blas_complex<T>;

// make_real
// metafunction to get a real type corresponding to some scalar (real or complex)
// Customize this for extension types
template <typename T> struct make_real;

template <real T> struct make_real<T>
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
  requires complex<T>
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
  requires numeric<T>
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
           (!numeric<T>) && numeric<typename numeric_type<typename T::value_type>::type> && (!detail::is_string<T>)
struct numeric_type<T> : numeric_type<typename T::value_type>
{};

template <typename T>
concept has_numeric_type = requires { typename numeric_type<T>::type; };

// scalar_type
// Recursively extracts the underlying scalar type from a container.
// For instance, for std::vector<std::vector<double>>, it returns double.

template <typename T> struct scalar_type;

template <typename T>
  requires scalar<T>
struct scalar_type<T>
{
    using type = T;
};

template <typename T>
  requires requires { typename T::value_type; } &&
           (!complex<T>) && scalar<typename scalar_type<typename T::value_type>::type>
struct scalar_type<T> : scalar_type<typename T::value_type>
{};

template <typename T>
concept has_scalar_type = requires { typename scalar_type<T>::type; };

} // namespace uni20
