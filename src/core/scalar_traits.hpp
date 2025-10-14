#pragma once

#include "types.hpp"
#include <complex>
#include <type_traits>

namespace uni20
{

/// \brief Trait to detect whether a type is an integer scalar (excluding `bool`).
/// \details Excludes `bool`, `char`, `signed char`, and `unsigned char`, which are often used for non-numeric data.
/// \note A customization point for user-defined integer-like scalars.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T>
struct is_integer_t : std::bool_constant<std::integral<T> && !std::same_as<T, bool> && !std::same_as<T, char> &&
                                         !std::same_as<T, signed char> && !std::same_as<T, unsigned char>>
{};

/// \brief Convenience alias that exposes the value of \c is_integer_t.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool is_integer = is_integer_t<std::remove_cvref_t<T>>::value;

/// \brief Trait to detect whether a type is a real-valued scalar.
/// \details Evaluates to true for `float`, `double`, or `long double`. Customize for other real scalar types as needed.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> struct is_real_t : std::false_type
{};

template <> struct is_real_t<float> : std::true_type
{};

template <> struct is_real_t<double> : std::true_type
{};

template <> struct is_real_t<long double> : std::true_type
{};

/// \brief Convenience alias that exposes the value of \c is_real_t.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool is_real = is_real_t<std::remove_cvref_t<T>>::value;

/// \brief Trait to detect whether a type is a complex scalar type.
/// \details Specializes for `std::complex<T>`; users may extend this for custom complex scalar wrappers.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> struct is_complex_t : std::false_type
{};

template <typename T> struct is_complex_t<std::complex<T>> : std::true_type
{};

/// \brief Convenience alias that exposes the value of \c is_complex_t.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool is_complex = is_complex_t<std::remove_cvref_t<T>>::value;

/// \brief Trait to detect whether a type is a numeric scalar (real, complex, or integer).
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool is_scalar = is_real<T> || is_complex<T> || is_integer<T>;

/// \brief Trait to detect whether a type is a real or complex scalar.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool is_real_or_complex = is_real<T> || is_complex<T>;

/// \brief Trait to extract the scalar type from a type `T`.
/// \details If `T` is a scalar (real, complex, or integer), returns `T`. If `T` is a container with a `value_type`
///          that itself names a scalar or another container, the trait recurses on `value_type`. When no scalar can
///          be located, `scalar_type<T>::type` aliases `void`.
/// \tparam T Type from which to extract a scalar component.
/// \ingroup core_math
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
/// \details Allows guarded use of `scalar_t<T>` in templates without triggering substitution failures.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool has_scalar = !std::same_as<scalar_t<T>, void>;

/// \brief Trait to detect whether `scalar_t<T>` is an integer scalar.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool has_integer_scalar = has_scalar<T> && is_integer<scalar_t<T>>;

/// \brief Trait to detect whether `scalar_t<T>` is a real scalar.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool has_real_scalar = has_scalar<T> && is_real<scalar_t<T>>;

/// \brief Trait to detect whether `scalar_t<T>` is a complex scalar.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool has_complex_scalar = has_scalar<T> && is_complex<scalar_t<T>>;

/// \brief Trait to detect whether a type has a scalar that is real or complex.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T>
inline constexpr bool has_real_or_complex_scalar = has_scalar<T> && is_real_or_complex<scalar_t<T>>;

/// \brief Metafunction to convert a type to its real-valued analog.
/// \details If `T` is a complex scalar, this returns the underlying real type. For real types, it returns `T`
///          unchanged.
/// \note This is a customization point. For containers like tensors, users can specialize this to
///       return a structurally identical container with real-valued elements.
/// \tparam T Type to convert.
/// \ingroup core_math
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
/// \tparam T Type to convert.
/// \ingroup core_math
template <typename T> using make_real_t = typename make_real<T>::type;

/// \brief Metafunction to convert a type to its complexified analog.
/// \details For real scalar types (e.g., `float`, `double`), returns `std::complex<T>`.
///          For types that already have a complex scalar (including containers), returns `T` unchanged.
/// \note This is a customization point. Users may specialize this for containers such as
///       tensors, enabling automatic transformation to complex-valued analogs.
/// \tparam T Type to convert.
/// \ingroup core_math
template <typename T> struct make_complex;

/// \brief Metafunction that produces the complex counterpart of a real-valued scalar.
/// \details For built-in floating point types this specialization aliases `std::complex<T>`.
/// \tparam T Floating-point type to complexify.
/// \ingroup core_math
template <typename T>
  requires std::floating_point<T>
struct make_complex<T>
{
    using type = std::complex<T>;
};

/// \brief Metafunction that leaves already-complex types unchanged.
/// \tparam T Type whose scalar component is already complex.
/// \ingroup core_math
template <typename T>
  requires has_complex_scalar<T>
struct make_complex<T>
{
    using type = T;
};

/// \brief Alias for the complexified version of `T`.
/// \tparam T Type to complexify.
/// \ingroup core_math
template <typename T> using make_complex_t = typename make_complex<T>::type;

} // namespace uni20
