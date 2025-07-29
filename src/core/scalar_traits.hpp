#pragma once

#include "types.hpp"
#include <complex>
#include <type_traits>

namespace uni20
{

/// \brief Trait to detect whether a type is an integer scalar (excluding `bool`).
///
/// Excludes `bool`, `char`, `signed char`, and `unsigned char`, which are often used for non-numeric data.
/// \note a customization point for user-defined integer types
template <typename T>
struct is_integer_t : std::bool_constant<std::integral<T> && !std::same_as<T, bool> && !std::same_as<T, char> &&
                                         !std::same_as<T, signed char> && !std::same_as<T, unsigned char>>
{};

template <typename T> inline constexpr bool is_integer = is_integer_t<std::remove_cvref_t<T>>::value;

/// \brief Trait to detect whether a type is a real-valued scalar.
///
/// Evaluates to true for `float`, `double`, or `long double`. Customize for
/// other types that act as real scalars
template <typename T> struct is_real_t : std::false_type
{};
template <> struct is_real_t<float> : std::true_type
{};
template <> struct is_real_t<double> : std::true_type
{};
template <> struct is_real_t<long double> : std::true_type
{};

template <typename T> inline constexpr bool is_real = is_real_t<std::remove_cvref_t<T>>::value;

/// \brief Trait to detect whether a type is a complex scalar type.
///
/// Specializes for `std::complex<T>`. Extend this for custom complex types.
template <typename T> struct is_complex_t : std::false_type
{};
template <typename T> struct is_complex_t<std::complex<T>> : std::true_type
{};

template <typename T> inline constexpr bool is_complex = is_complex_t<std::remove_cvref_t<T>>::value;

/// \brief Trait to detect whether a type is a numeric scalar (real, complex, or integer).
template <typename T> inline constexpr bool is_scalar = is_real<T> || is_complex<T> || is_integer<T>;

/// \brief Trait to detect whether a type is a real or complex scalar.
template <typename T> inline constexpr bool is_real_or_complex = is_real<T> || is_complex<T>;

/// \brief Trait to extract the scalar type from a type `T`.
///
/// If `T` is a scalar (real, complex, or integer), returns `T`.
/// If `T` is a container with a `value_type` that is, or contains a scalar_type, recurses on that type.
/// If 'T' does not contain a scalar type, `scalar_type<T>::type` is `void`.
template <typename T> struct scalar_type
{
    using type = void;
};

template <typename T>
  requires is_scalar<T>
struct scalar_type<T>
{
    using type = T;
};

template <typename T>
  requires requires { typename T::value_type; } && (!is_scalar<T>) &&
           requires { typename scalar_type<typename T::value_type>::type; }
struct scalar_type<T> : scalar_type<typename T::value_type>
{};

template <typename T> using scalar_t = typename scalar_type<std::remove_cvref_t<T>>::type;

/// \brief Trait to detect whether `scalar_t<T>` is well-formed.
///
/// Allows guarded use of `scalar_t<T>` in templates without triggering substitution failures.
// template <typename T>
// inline constexpr bool has_scalar = requires { typename scalar_type<std::remove_cvref_t<T>>::type; };
template <typename T> inline constexpr bool has_scalar = !std::same_as<scalar_t<T>, void>;

/// \brief Trait to detect whether `scalar_t<T>` is an integer scalar.
template <typename T> inline constexpr bool has_integer_scalar = has_scalar<T> && is_integer<scalar_t<T>>;

/// \brief Trait to detect whether `scalar_t<T>` is a real scalar.
template <typename T> inline constexpr bool has_real_scalar = has_scalar<T> && is_real<scalar_t<T>>;

/// \brief Trait to detect whether `scalar_t<T>` is a complex scalar.
template <typename T> inline constexpr bool has_complex_scalar = has_scalar<T> && is_complex<scalar_t<T>>;

/// \brief Trait to detect whether a type has a scalar that is real or complex.
template <typename T>
inline constexpr bool has_real_or_complex_scalar = has_scalar<T> && is_real_or_complex<scalar_t<T>>;

/// \brief Metafunction to convert a type to its real-valued analog.
///
/// If `T` is a complex scalar, this returns the underlying real type.
/// For real types, it returns `T` unchanged.
/// \note This is a customization point. For containers like tensors, users can specialize this
///       to return a structurally identical container with real-valued elements.
template <typename T> struct make_real;

template <typename T>
  requires has_real_scalar<T>
struct make_real<T>
{
    using type = T;
};

template <typename T>
  requires is_complex<T>
struct make_real<T>
{
    using type = typename T::value_type;
};

/// \brief Alias for the underlying real-valued type of `T`.
template <typename T> using make_real_t = typename make_real<T>::type;

/// \brief Metafunction to convert a type to its complexified analog.
///
/// For real scalar types (e.g., `float`, `double`), returns `std::complex<T>`.
/// For types that already have a complex scalar (including containers), returns `T` unchanged.
///
/// \note This is a customization point. Users may specialize this for containers such as
///       tensors, enabling automatic transformation to complex-valued analogs.
template <typename T> struct make_complex;

/// For built-in floating point types the complex_type is std::complex<T>
template <typename T>
  requires std::floating_point<T>
struct make_complex<T>
{
    using type = std::complex<T>;
};

template <typename T>
  requires has_complex_scalar<T>
struct make_complex<T>
{
    using type = T;
};

/// \brief Alias for the complexified version of `T`.
template <typename T> using make_complex_t = typename make_complex<T>::type;

} // namespace uni20
