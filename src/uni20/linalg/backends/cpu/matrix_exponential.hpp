/**
 * \file matrix_exponential.hpp
 * \brief CPU dense matrix exponential interface.
 */

#pragma once

#include <uni20/config.hpp>

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
#include <mplapack_config.h>
#endif
#include <complex>

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backends/cpu/dense_matrix.hpp>

namespace uni20::linalg::backends::cpu
{

/// \brief Compute the matrix exponential using the adaptive scaling-and-squaring algorithm.
/// \details Follows the Pad\'e-based scaling and squaring strategy of Higham (2005) and
///          Al-Mohy & Higham (2011). The routine automatically selects between Pad\'e
///          degrees {3, 5, 7, 9, 13} based on matrix norms.
/// \tparam Scalar Dense matrix element type; may be real or complex in single, double, or extended precision.
/// \param matrix Dense matrix whose exponential will be evaluated.
/// \param t Scalar multiplier applied to \p matrix before exponentiation.
/// \return The matrix exponential of \f$\exp(t \cdot \text{matrix})\f$.
template <uni20::RealOrComplex Scalar>
DenseMatrix<Scalar> matrix_exponential(DenseMatrix<Scalar> const& matrix, uni20::make_real_t<Scalar> t);

/// \brief Compute a complex-coefficient matrix exponential, promoting real matrices to complex output.
/// \details This overload evaluates \f$\exp(t A)\f$ for complex \p t. If \p matrix is real, the result
///          is promoted to a complex dense matrix. If \p matrix is already complex, the result remains complex.
/// \tparam Scalar Real or complex matrix element type.
/// \param matrix Dense matrix whose exponential will be evaluated.
/// \param t Complex multiplier applied to \p matrix before exponentiation.
/// \return The complex matrix exponential of \f$\exp(t \cdot \text{matrix})\f$.
template <uni20::RealOrComplex Scalar>
DenseMatrix<uni20::complex<uni20::make_real_t<Scalar>>>
matrix_exponential(DenseMatrix<Scalar> const& matrix, uni20::complex<uni20::make_real_t<Scalar>> t);

} // namespace uni20::linalg::backends::cpu
