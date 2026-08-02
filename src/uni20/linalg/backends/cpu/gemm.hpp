#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Reference CPU GEMM backend for tensor-view operands.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/cpu/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/matrix_product_shape.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/output.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class OutputMdspan, class LhsMdspan, class RhsMdspan>
concept HostGemmMdspanAccess = uni20::HostWritableMdspec<OutputMdspan> && uni20::HostReadableMdspec<LhsMdspan> &&
                               uni20::HostReadableMdspec<RhsMdspan>;

template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
consteval bool cpu_gemm_mdspan_types_compatible()
{
  using output_span = uni20::host_write_mdspan_t<OutputMdspan>;
  using lhs_span = uni20::host_read_mdspan_t<LhsMdspan>;
  using rhs_span = uni20::host_read_mdspan_t<RhsMdspan>;
  return uni20::linalg::cpu::GemmCompatible<output_span, Scalar, lhs_span, rhs_span>;
}

template <class OutputTensor, class LhsMdspan, class RhsMdspan>
concept HostAssignProductAccess =
    HostGemmMdspanAccess<std::remove_cvref_t<decltype(uni20::mdspec_of(std::declval<OutputTensor&>()))>, LhsMdspan,
                         RhsMdspan>;

template <class OutputTensor, class Scalar, class LhsMdspan, class RhsMdspan>
consteval bool cpu_assign_product_types_compatible()
{
  using output_span = std::remove_cvref_t<decltype(uni20::mdspec_of(std::declval<OutputTensor&>()))>;
  return cpu_gemm_mdspan_types_compatible<output_span, Scalar, LhsMdspan, RhsMdspan>();
}

template <uni20::MutableRankedMdspecLike<2> OutputMdspan, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires HostGemmMdspanAccess<OutputMdspan, LhsMdspan, RhsMdspan>
KernelAttempt try_cpu_gemm(OutputMdspan& output, Scalar alpha, LhsMdspan& lhs, RhsMdspan& rhs, Scalar beta)
{
  auto output_access = acquire_host_write_access_sync(output);
  auto lhs_access = acquire_host_read_access_sync(lhs);
  auto rhs_access = acquire_host_read_access_sync(rhs);
  auto output_span = output_access.mdspan();
  auto lhs_span = lhs_access.mdspan();
  auto rhs_span = rhs_access.mdspan();
  uni20::linalg::cpu::gemm(output_span, alpha, lhs_span, rhs_span, beta);
  return KernelAttempt::success;
}
} // namespace detail

/// \brief Report eligibility for host-accessible mdspec CPU GEMM.
template <uni20::MutableRankedMdspecLike<2> OutputMdspan, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::HostGemmMdspanAccess<OutputMdspan, LhsMdspan, RhsMdspan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, gemm_op const&, OutputMdspan&, Scalar const&,
                                    LhsMdspan&, RhsMdspan&, Scalar const&)
{
  if constexpr (detail::cpu_gemm_mdspan_types_compatible<OutputMdspan, Scalar, LhsMdspan, RhsMdspan>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host access and run reference GEMM.
template <uni20::MutableRankedMdspecLike<2> OutputMdspan, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::HostGemmMdspanAccess<OutputMdspan, LhsMdspan, RhsMdspan>
KernelAttempt try_kernel(CpuReferenceBackend, gemm_op const&, OutputMdspan& output, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs, Scalar beta)
{
  return detail::try_cpu_gemm(output, alpha, lhs, rhs, beta);
}

/// \brief Report eligibility for replaceable-output host tensor matrix products.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::HostAssignProductAccess<OutputTensor, LhsMdspan, RhsMdspan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, assign_product_op const&, OutputTensor&, Scalar const&,
                                    LhsMdspan&, RhsMdspan&)
{
  if constexpr (detail::cpu_assign_product_types_compatible<OutputTensor, Scalar, LhsMdspan, RhsMdspan>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Lower a replaceable-output host tensor matrix product to fixed-output GEMM.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::HostAssignProductAccess<OutputTensor, LhsMdspan, RhsMdspan>
KernelAttempt try_kernel(CpuReferenceBackend, assign_product_op const&, OutputTensor& output, Scalar alpha,
                         LhsMdspan& lhs, RhsMdspan& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  uni20::prepare_output(output, shape);
  auto output_span = uni20::mdspec_of(output);
  return detail::try_cpu_gemm(output_span, alpha, lhs, rhs, Scalar{});
}

/// \brief Report eligibility for a deferred host Tensor matrix-product output.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::HostAssignProductAccess<OutputTensor, LhsMdspan, RhsMdspan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, assign_product_op const&,
                                    async::shared_storage<OutputTensor>&, Scalar const&, LhsMdspan&, RhsMdspan&)
{
  if constexpr (detail::cpu_assign_product_types_compatible<OutputTensor, Scalar, LhsMdspan, RhsMdspan>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Construct or resize a deferred host output and run reference GEMM.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::HostAssignProductAccess<OutputTensor, LhsMdspan, RhsMdspan>
KernelAttempt try_kernel(CpuReferenceBackend, assign_product_op const&,
                         async::shared_storage<OutputTensor>& output_storage, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  auto& output = uni20::prepare_output(output_storage, shape);
  auto output_span = uni20::mdspec_of(output);
  return detail::try_cpu_gemm(output_span, alpha, lhs, rhs, Scalar{});
}

} // namespace uni20::linalg
