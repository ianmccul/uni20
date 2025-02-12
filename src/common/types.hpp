#pragma once

#include "config.hpp"
#include <complex>
#include <concepts>

namespace uni20
{

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

//
// Concepts
//
// integral       - integral type (same as std::integral)
// real           - a real floating point number (including extensions that are floating-point-like)
// complex        - complex floating point; either std::complex<RealType> or some complex-like extension
// scalar         - either real or complex
// blas_real      - real types accepted by standard BLAS (i.e. single and double precision)
// blas_complex   - complex types accepted by standard BLAS (i.e. single and double precision complex)
// blas_scalar    - union of blas_real and blas_complex
//

using std::integral;

template <typename T>
concept real = is_real<T>;

template <typename T>
concept complex = is_complex<T>;

template <typename T>
concept scalar = real<T> || complex<T>;

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
  requires complex<T>
struct scalar_type<T>
{
    using type = T;
};

template <typename T>
  requires requires { typename T::value_type; } && (!complex<T>)
struct scalar_type<T> : scalar_type<typename T::value_type>
{};

} // namespace uni20
