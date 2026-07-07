#pragma once

#include "numeric_limits.hpp"
#include "scalar_concepts.hpp"
#include <complex>
#include <numeric>
#include <type_traits>

namespace uni20
{

/**
 * \brief Scalar math helper utilities.
 *
 * \file math.hpp
 * \ingroup core
 */

/**
 * \brief Scalar helper utilities shared across Uni20 core algorithms.
 *
 * \defgroup core_math Scalar helper utilities
 * \ingroup core
 */

/// \brief Indicates whether the Uni20 conjugation helper is a no-op for the provided scalar type.
/// \details Evaluates to `true` when the scalar is already real-valued or integral, allowing callers to skip
///         complex conjugation work. The variable template is `constexpr`, so the result may be used in
///         constant-expression contexts.
/// \tparam T Scalar type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool has_trivial_conj = has_real_scalar_v<T> || has_integer_scalar_v<T>;

/// \brief Returns the complex conjugate for complex-valued scalars.
/// \details This overload forwards to `std::conj` and therefore returns a `uni20::complex<T>` copy of the
///         input value. It inherits the constexpr availability of `std::conj` (currently not `constexpr`).
/// \tparam T Component type of the complex scalar.
/// \param x Complex value whose conjugate is requested.
/// \return The complex conjugate of `x`.
/// \ingroup core_math
template <typename T> uni20::complex<T> conj(uni20::complex<T> x) { return std::conj(x); }

/// \brief Returns the conjugate of a real-valued scalar.
/// \details Real numbers are unchanged by conjugation, so the value is returned verbatim. The overload is
///         `constexpr`, enabling compile-time evaluation for literal arguments.
/// \tparam R Real scalar type.
/// \param x Real scalar to return.
/// \return `x`, unchanged.
/// \ingroup core_math
template <Real R> constexpr R conj(R const& x) { return x; }

/// \brief Returns the conjugate of an integer scalar.
/// \details Integer values are treated as reals for conjugation and therefore returned unchanged. The
///         overload is `constexpr`, enabling compile-time evaluation for literal arguments.
/// \tparam I Integer scalar type.
/// \param x Integer scalar to return.
/// \return `x`, unchanged.
/// \ingroup core_math
template <Integer I> constexpr I conj(I const& x) { return x; }

/// \brief Computes the Hermitian adjoint of a scalar value.
/// \details For scalar inputs the Hermitian adjoint is equivalent to the complex conjugate, so this helper
/// simply forwards to `conj`. When the selected `conj` overload is `constexpr`, this helper is as
/// well, preserving compile-time evaluation.
/// \tparam S Scalar type satisfying \c HasScalar.
/// \param x Scalar value whose Hermitian adjoint is requested.
/// \return The result of calling `conj(x)`.
/// \ingroup core_math
template <HasScalar S> constexpr auto herm(S x) { return uni20::conj(x); }

namespace detail
{
template <typename T, typename = void> struct numeric_limits_has_infinity : std::false_type
{};

template <typename T>
struct numeric_limits_has_infinity<
    T, std::void_t<decltype(uni20::numeric_limits<T>::has_infinity), decltype(uni20::numeric_limits<T>::infinity())>>
    : std::bool_constant<static_cast<bool>(uni20::numeric_limits<T>::has_infinity)>
{};

template <typename T>
inline constexpr bool numeric_limits_has_infinity_v = numeric_limits_has_infinity<std::remove_cvref_t<T>>::value;
} // namespace detail

/// \brief Returns whether an integer scalar is finite.
/// \details Integer Uni20 scalars have no NaN or infinity representation, so every value is finite.
/// \tparam I Integer scalar type.
/// \param x Integer scalar to inspect.
/// \return Always `true`.
/// \ingroup core_math
template <Integer I> constexpr bool isfinite(I const& x) noexcept
{
  (void)x;
  return true;
}

/// \brief Returns whether a real scalar is neither NaN nor positive/negative infinity.
/// \details This uses `uni20::numeric_limits<T>` so extension real scalar types can define their own infinity
///         representation without depending on standard-library overload coverage for `std::isfinite`.
/// \tparam R Real scalar type.
/// \param x Real scalar to inspect.
/// \return `true` when `x` is finite.
/// \ingroup core_math
template <Real R> constexpr bool isfinite(R const& x)
{
  using value_type = std::remove_cvref_t<R>;
  if (!(x == x))
  {
    return false;
  }
  if constexpr (detail::numeric_limits_has_infinity_v<value_type>)
  {
    auto const infinity = uni20::numeric_limits<value_type>::infinity();
    return x != infinity && x != -infinity;
  }
  else
  {
    return true;
  }
}

/// \brief Returns whether both components of a complex scalar are finite.
/// \tparam T Real component type.
/// \param z Complex scalar to inspect.
/// \return `true` when both the real and imaginary components are finite.
/// \ingroup core_math
template <typename T> constexpr bool isfinite(uni20::complex<T> const& z)
{
  return uni20::isfinite(z.real()) && uni20::isfinite(z.imag());
}

/// \brief Provides mutable access to the real component of a `uni20::complex` value.
/// \details This helper mirrors the `std::real` overload for lvalues while remaining `constexpr` and
/// `noexcept` for direct reference access.
/// \tparam T Component type of the complex scalar.
/// \param z Complex number whose real component will be exposed.
/// \return Reference to the real component of `z`.
/// \ingroup core_math
template <typename T> constexpr T& real(uni20::complex<T>& z) noexcept { return reinterpret_cast<T*>(&z)[0]; }

using std::real;

/// \brief Provides mutable access to the imaginary component of a `uni20::complex` value.
/// \details This helper mirrors the `std::imag` overload for lvalues while remaining `constexpr` and
/// `noexcept` for direct reference access.
/// \tparam T Component type of the complex scalar.
/// \param z Complex number whose imaginary component will be exposed.
/// \return Reference to the imaginary component of `z`.
/// \ingroup core_math
template <typename T> constexpr T& imag(uni20::complex<T>& z) noexcept { return reinterpret_cast<T*>(&z)[1]; }

using std::imag;

} // namespace uni20
