#pragma once

#include <uni20/linalg/dispatch_diagnostics.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/linalg/ops/gemv.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/lq.hpp>
#include <uni20/linalg/ops/matrix_exponential.hpp>
#include <uni20/linalg/ops/matrix_norm.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/linalg/ops/matrix_set.hpp>
#include <uni20/linalg/ops/nonsymmetric_eigen.hpp>
#include <uni20/linalg/ops/qr.hpp>
#include <uni20/linalg/ops/schur.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/linalg/ops/tridiagonal_eigen.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>
#include <uni20/tensor/conjugate_inplace.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/reductions.hpp>
#include <uni20/tensor/transform.hpp>

namespace uni20::linalg
{
/// \brief Entry point header for Uni20 linear algebra facilities.
} // namespace uni20::linalg
