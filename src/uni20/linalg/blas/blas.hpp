#pragma once

/**
 * \file blas.hpp
 * \ingroup linalg
 * \brief Include point for mdspan-to-BLAS dense linalg adapters.
 */

#include <uni20/linalg/blas/blas_matrix.hpp>
#include <uni20/linalg/blas/blas_vector.hpp>
#include <uni20/linalg/blas/contract.hpp>
#include <uni20/linalg/blas/gemm.hpp>
#include <uni20/linalg/blas/gemv.hpp>
#include <uni20/linalg/blas/mdspan_access.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/blas/mdspan_vector.hpp>

namespace uni20::linalg::blas
{} // namespace uni20::linalg::blas
