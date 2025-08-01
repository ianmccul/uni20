#pragma once

#include "scalar_traits.hpp"
#include "types.hpp"

namespace uni20
{

/// \brief Concept for integer scalar types (excluding char and bool).
template <typename T>
concept Integer = is_integer<T>;

/// \brief Concept for real scalar types (float, double, etc.).
template <typename T>
concept Real = is_real<T>;

/// \brief Concept for complex scalar types.
template <typename T>
concept Complex = is_complex<T>;

/// \brief Concept for numeric scalar types (integer, real, or complex).
template <typename T>
concept Scalar = is_scalar<T>;

/// \brief Concept for types that are either real or complex.
template <typename T>
concept RealOrComplex = Real<T> || Complex<T>;

/// \brief Concept for BLAS-compatible real scalar types.
template <typename T>
concept BlasReal = std::same_as<T, float> || std::same_as<T, double>;

/// \brief Concept for BLAS-compatible complex scalar types.
template <typename T>
concept BlasComplex = std::same_as<T, cfloat> || std::same_as<T, cdouble>;

/// \brief Concept for all BLAS-compatible scalar types.
template <typename T>
concept BlasScalar = BlasReal<T> || BlasComplex<T>;

/// \brief Concept for a type that is a scalar, or has a scalar value_type
template <typename T>
concept HasScalar = has_scalar<T>;

/// \brief Concept for types whose `scalar_t<T>` is an integer.
template <typename T>
concept HasIntegerScalar = has_integer_scalar<T>;

/// \brief Concept for types whose `scalar_t<T>` is real.
template <typename T>
concept HasRealScalar = has_real_scalar<T>;

/// \brief Concept for types whose `scalar_t<T>` is complex.
template <typename T>
concept HasComplexScalar = has_complex_scalar<T>;

} // namespace uni20
