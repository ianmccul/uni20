#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Operation-tag GEMM front end for dense mdspan-like matrices.
 */

#include <uni20/linalg/backends/cpu/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/tensor/tensor_view.hpp>

#if UNI20_BACKEND_BLAS
#include <uni20/linalg/backends/blas/gemm.hpp>
#endif

#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Dense matrix multiplication operation tag.
struct gemm_op
{};

/// \brief Try `output = alpha * lhs * rhs + beta * output` through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedSpanLike<2> OutputMdspan, class Scalar,
          uni20::RankedSpanLike<2> LhsMdspan, uni20::RankedSpanLike<2> RhsMdspan>
bool try_gemm(BackendSelector&& selector, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs,
              Scalar beta)
{
  auto backends = normalize_backend_selector(std::forward<BackendSelector>(selector));
  using backends_type = std::remove_cvref_t<decltype(backends)>;
  constexpr auto acceptance = probe_dispatch_kernel(
      detail::kernel_type_probe_arg<backends_type const&>(), detail::kernel_type_probe_arg<gemm_op const&>(),
      detail::kernel_type_probe_arg<OutputMdspan&&>(), detail::kernel_type_probe_arg<Scalar&>(),
      detail::kernel_type_probe_arg<LhsMdspan&&>(), detail::kernel_type_probe_arg<RhsMdspan&&>(),
      detail::kernel_type_probe_arg<Scalar&>());
  static_assert(acceptance != KernelTypeAcceptance::no,
                "no backend in this selector can ever service gemm for these argument types");

  return try_dispatch_kernel(backends, gemm_op{}, std::forward<OutputMdspan>(output), alpha,
                             std::forward<LhsMdspan>(lhs), std::forward<RhsMdspan>(rhs), beta);
}

/// \brief Checked `output = alpha * lhs * rhs + beta * output` through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedSpanLike<2> OutputMdspan, class Scalar,
          uni20::RankedSpanLike<2> LhsMdspan, uni20::RankedSpanLike<2> RhsMdspan>
void gemm(BackendSelector&& selector, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs,
          Scalar beta)
{
  auto backends = normalize_backend_selector(std::forward<BackendSelector>(selector));
  using backends_type = std::remove_cvref_t<decltype(backends)>;
  constexpr auto acceptance = probe_dispatch_kernel(
      detail::kernel_type_probe_arg<backends_type const&>(), detail::kernel_type_probe_arg<gemm_op const&>(),
      detail::kernel_type_probe_arg<OutputMdspan&&>(), detail::kernel_type_probe_arg<Scalar&>(),
      detail::kernel_type_probe_arg<LhsMdspan&&>(), detail::kernel_type_probe_arg<RhsMdspan&&>(),
      detail::kernel_type_probe_arg<Scalar&>());
  static_assert(acceptance != KernelTypeAcceptance::no,
                "no backend in this selector can ever service gemm for these argument types");

  dispatch_kernel(backends, gemm_op{}, std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                  std::forward<RhsMdspan>(rhs), beta);
}

/// \brief Try fixed-storage tensor GEMM through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, class Scalar,
          uni20::RankedTensorView<2> LhsTensor, uni20::RankedTensorView<2> RhsTensor>
bool try_gemm(BackendSelector&& selector, OutputTensor&& output, Scalar alpha, LhsTensor const& lhs,
              RhsTensor const& rhs, Scalar beta)
{
  return try_gemm(std::forward<BackendSelector>(selector), output.mdspan(), alpha, lhs.mdspan(), rhs.mdspan(), beta);
}

/// \brief Run fixed-storage tensor GEMM through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, class Scalar,
          uni20::RankedTensorView<2> LhsTensor, uni20::RankedTensorView<2> RhsTensor>
void gemm(BackendSelector&& selector, OutputTensor&& output, Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs,
          Scalar beta)
{
  gemm(std::forward<BackendSelector>(selector), output.mdspan(), alpha, lhs.mdspan(), rhs.mdspan(), beta);
}

namespace detail
{
template <class OutputTensor, class LhsTensor, class RhsTensor>
[[nodiscard]] constexpr auto common_tensor_backend_selector(OutputTensor const& output, LhsTensor const& lhs,
                                                            RhsTensor const& rhs)
{
  using output_selector = std::remove_cvref_t<decltype(output.backend_selector())>;
  using lhs_selector = std::remove_cvref_t<decltype(lhs.backend_selector())>;
  using rhs_selector = std::remove_cvref_t<decltype(rhs.backend_selector())>;
  static_assert(std::same_as<output_selector, lhs_selector> && std::same_as<output_selector, rhs_selector>,
                "tensor GEMM operands must provide the same default backend selector type");
  return output.backend_selector();
}
} // namespace detail

/// \brief Try fixed-storage tensor GEMM using the operands' default backend selector.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
bool try_gemm(OutputTensor&& output, Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs, Scalar beta)
{
  auto selector = detail::common_tensor_backend_selector(output, lhs, rhs);
  return try_gemm(selector, std::forward<OutputTensor>(output), alpha, lhs, rhs, beta);
}

/// \brief Run fixed-storage tensor GEMM using the operands' default backend selector.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
void gemm(OutputTensor&& output, Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs, Scalar beta)
{
  auto selector = detail::common_tensor_backend_selector(output, lhs, rhs);
  gemm(selector, std::forward<OutputTensor>(output), alpha, lhs, rhs, beta);
}

} // namespace uni20::linalg
