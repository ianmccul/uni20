#pragma once

#include "scalar_concepts.hpp"
#include <complex>
#include <numeric>

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
template <typename T> inline constexpr bool has_trivial_conj = has_real_scalar<T> || has_integer_scalar<T>;

/// \brief Returns the complex conjugate for complex-valued scalars.
/// \details This overload forwards to `std::conj` and therefore returns a `std::complex<T>` copy of the
///         input value. It inherits the constexpr availability of `std::conj` (currently not `constexpr`).
/// \tparam T Component type of the complex scalar.
/// \param x Complex value whose conjugate is requested.
/// \return The complex conjugate of `x`.
/// \ingroup core_math
template <typename T> std::complex<T> conj(std::complex<T> x) { return std::conj(x); }

/// \brief Returns the conjugate of a real-valued scalar.
/// \details Real numbers are unchanged by conjugation, so the value is returned verbatim. The overload is
///         `constexpr`, enabling compile-time evaluation for literal arguments.
/// \tparam R Real scalar type that satisfies \c HasRealScalar.
/// \param x Real scalar to return.
/// \return `x`, unchanged.
/// \ingroup core_math
template <HasRealScalar R> constexpr R conj(R const& x) { return x; }

/// \brief Returns the conjugate of an integer scalar.
/// \details Integer values are treated as reals for conjugation and therefore returned unchanged. The
///         overload is `constexpr`, enabling compile-time evaluation for literal arguments.
/// \tparam I Integer scalar type that satisfies \c HasIntegerScalar.
/// \param x Integer scalar to return.
/// \return `x`, unchanged.
/// \ingroup core_math
template <HasIntegerScalar I> constexpr I conj(I const& x) { return x; }

/// \brief Computes the Hermitian adjoint of a scalar value.
/// \details For scalar inputs the Hermitian adjoint is equivalent to the complex conjugate, so this helper
/// simply forwards to `conj`. When the selected `conj` overload is `constexpr`, this helper is as
/// well, preserving compile-time evaluation.
/// \tparam S Scalar type satisfying \c HasScalar.
/// \param x Scalar value whose Hermitian adjoint is requested.
/// \return The result of calling `conj(x)`.
/// \ingroup core_math
template <HasScalar S> constexpr auto herm(S x) { return conj(x); }

/// \brief Provides mutable access to the real component of a `std::complex` value.
/// \details This helper mirrors the `std::real` overload for lvalues while remaining `constexpr` and
/// `noexcept` for direct reference access.
/// \tparam T Component type of the complex scalar.
/// \param z Complex number whose real component will be exposed.
/// \return Reference to the real component of `z`.
/// \ingroup core_math
template <typename T> constexpr T& real(std::complex<T>& z) noexcept { return reinterpret_cast<T*>(&z)[0]; }

using std::real;

/// \brief Provides mutable access to the imaginary component of a `std::complex` value.
/// \details This helper mirrors the `std::imag` overload for lvalues while remaining `constexpr` and
/// `noexcept` for direct reference access.
/// \tparam T Component type of the complex scalar.
/// \param z Complex number whose imaginary component will be exposed.
/// \return Reference to the imaginary component of `z`.
/// \ingroup core_math
template <typename T> constexpr T& imag(std::complex<T>& z) noexcept { return reinterpret_cast<T*>(&z)[1]; }

using std::imag;

} // namespace uni20
