#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Operation-tag GEMM front end for dense mdspan-like matrices.
 */

#include <uni20/linalg/backends/cpu/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>

#if UNI20_BACKEND_BLAS
#include <uni20/linalg/backends/blas/gemm.hpp>
#endif

#include <utility>

namespace uni20::linalg
{

/// \brief Dense matrix multiplication operation tag.
struct gemm_op
{};

/// \brief Try `output = alpha * lhs * rhs + beta * output` through an explicit backend selector.
template <class BackendSelector, class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
bool try_gemm(BackendSelector&& selector, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs,
              Scalar beta)
{
  auto backends = normalize_backend_selector(std::forward<BackendSelector>(selector));
  using backends_type = std::remove_cvref_t<decltype(backends)>;
  static_assert(
      any_kernel_type_eligible_v<backends_type, gemm_op, OutputMdspan&&, Scalar&, LhsMdspan&&, RhsMdspan&&, Scalar&>,
      "no backend in this selector can ever service gemm for these argument types");

  return dispatch_kernel(backends, gemm_op{}, std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                         std::forward<RhsMdspan>(rhs), beta);
}

/// \brief Checked `output = alpha * lhs * rhs + beta * output` through an explicit backend selector.
template <class BackendSelector, class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
void gemm(BackendSelector&& selector, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs,
          Scalar beta)
{
  CHECK(try_gemm(std::forward<BackendSelector>(selector), std::forward<OutputMdspan>(output), alpha,
                 std::forward<LhsMdspan>(lhs), std::forward<RhsMdspan>(rhs), beta));
}

} // namespace uni20::linalg
