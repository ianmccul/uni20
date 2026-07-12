#pragma once

/**
 * \file blas_vector.hpp
 * \ingroup linalg
 * \brief Provider-ready BLAS vector operands.
 */

#include <uni20/config.hpp>

namespace uni20::linalg::blas
{

/// \brief Writable vector shape as seen by a BLAS provider.
template <class Scalar, class Handle = Scalar*> struct BlasWritableVector
{
    Handle data{};
    blas_int size = 0;
    blas_int increment = 1;
};

/// \brief Readable vector shape as seen by a BLAS provider.
template <class Scalar, class Handle = Scalar const*> struct BlasReadableVector
{
    Handle data{};
    blas_int size = 0;
    blas_int increment = 1;
};

} // namespace uni20::linalg::blas
