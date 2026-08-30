#pragma once

/**
 * \file async.hpp
 * \ingroup linalg
 * \brief Opt-in asynchronous Tensor wrappers for dense linear algebra operations.
 */

#include <uni20/linalg/async/concepts.hpp>
#include <uni20/linalg/async/conjugate_inplace.hpp>
#include <uni20/linalg/async/contract.hpp>
#include <uni20/linalg/async/copy.hpp>
#include <uni20/linalg/async/gemv.hpp>
#include <uni20/linalg/async/linear_solve.hpp>
#include <uni20/linalg/async/lq.hpp>
#include <uni20/linalg/async/matrix_exponential.hpp>
#include <uni20/linalg/async/matrix_norm.hpp>
#include <uni20/linalg/async/matrix_product.hpp>
#include <uni20/linalg/async/matrix_set.hpp>
#include <uni20/linalg/async/qr.hpp>
#include <uni20/linalg/async/reductions.hpp>
#include <uni20/linalg/async/self_adjoint_eigh.hpp>
#include <uni20/linalg/async/svd.hpp>
#include <uni20/linalg/async/transform.hpp>
#include <uni20/linalg/async/truncated_svd.hpp>

namespace uni20::linalg
{
/// \brief Entry point header for asynchronous Tensor linalg operations.
} // namespace uni20::linalg
