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
concept HostGemmMdspanAccess =
    uni20::detail::HostWritableDeviceMdspan<OutputMdspan> && uni20::detail::HostReadableDeviceMdspan<LhsMdspan> &&
    uni20::detail::HostReadableDeviceMdspan<RhsMdspan>;

template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
consteval bool cpu_gemm_mdspan_types_compatible()
{
  using output_span = uni20::detail::host_write_mdspan_t<OutputMdspan>;
  using lhs_span = uni20::detail::host_read_mdspan_t<LhsMdspan>;
  using rhs_span = uni20::detail::host_read_mdspan_t<RhsMdspan>;
  return uni20::linalg::cpu::GemmCompatible<output_span, Scalar, lhs_span, rhs_span>;
}

template <class OutputTensor, class LhsTensor, class RhsTensor>
concept HostGemmTensorAccess = HostGemmMdspanAccess<
    std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<OutputTensor&>()))>,
    std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<LhsTensor const&>()))>,
    std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<RhsTensor const&>()))>>;

template <class OutputTensor, class Scalar, class LhsTensor, class RhsTensor> consteval bool cpu_gemm_types_compatible()
{
  using output_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<OutputTensor&>()))>;
  using lhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<LhsTensor const&>()))>;
  using rhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<RhsTensor const&>()))>;
  return cpu_gemm_mdspan_types_compatible<output_span, Scalar, lhs_span, rhs_span>();
}

template <uni20::MutableRankedDeviceMdspanLike<2> OutputMdspan, uni20::Scalar Scalar,
          uni20::RankedDeviceMdspanLike<2> LhsMdspan, uni20::RankedDeviceMdspanLike<2> RhsMdspan>
  requires HostGemmMdspanAccess<OutputMdspan, LhsMdspan, RhsMdspan>
KernelAttempt try_cpu_gemm(OutputMdspan& output, Scalar alpha, LhsMdspan& lhs, RhsMdspan& rhs, Scalar beta)
{
  auto output_access = acquire_host_write_access(output);
  auto lhs_access = acquire_host_read_access(lhs);
  auto rhs_access = acquire_host_read_access(rhs);
  auto output_span = output_access.mdspan();
  auto lhs_span = lhs_access.mdspan();
  auto rhs_span = rhs_access.mdspan();
  uni20::linalg::cpu::gemm(output_span, alpha, lhs_span, rhs_span, beta);
  return KernelAttempt::success;
}
} // namespace detail

/// \brief Report eligibility for host-accessible device-mdspan CPU GEMM.
template <uni20::MutableRankedDeviceMdspanLike<2> OutputMdspan, uni20::Scalar Scalar,
          uni20::RankedDeviceMdspanLike<2> LhsMdspan, uni20::RankedDeviceMdspanLike<2> RhsMdspan>
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
template <uni20::MutableRankedDeviceMdspanLike<2> OutputMdspan, uni20::Scalar Scalar,
          uni20::RankedDeviceMdspanLike<2> LhsMdspan, uni20::RankedDeviceMdspanLike<2> RhsMdspan>
  requires detail::HostGemmMdspanAccess<OutputMdspan, LhsMdspan, RhsMdspan>
KernelAttempt try_kernel(CpuReferenceBackend, gemm_op const&, OutputMdspan& output, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs, Scalar beta)
{
  return detail::try_cpu_gemm(output, alpha, lhs, rhs, beta);
}

/// \brief Report eligibility for replaceable-output host tensor matrix products.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::Scalar Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::HostGemmTensorAccess<OutputTensor, LhsTensor, RhsTensor>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, assign_product_op const&, OutputTensor&, Scalar const&,
                                    LhsTensor&, RhsTensor&)
{
  if constexpr (detail::cpu_gemm_types_compatible<OutputTensor, Scalar, LhsTensor, RhsTensor>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Lower a replaceable-output host tensor matrix product to fixed-output GEMM.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::Scalar Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::HostGemmTensorAccess<OutputTensor, LhsTensor, RhsTensor>
KernelAttempt try_kernel(CpuReferenceBackend, assign_product_op const&, OutputTensor& output, Scalar alpha,
                         LhsTensor const& lhs, RhsTensor const& rhs)
{
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  auto const shape = detail::matrix_product_shape(lhs_span, rhs_span);
  uni20::prepare_output(output, shape);
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  return detail::try_cpu_gemm(output_span, alpha, lhs_span, rhs_span, Scalar{});
}

/// \brief Report eligibility for a deferred host Tensor matrix-product output.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::Scalar Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::HostGemmTensorAccess<OutputTensor, LhsTensor, RhsTensor>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, assign_product_op const&,
                                    async::shared_storage<OutputTensor>&, Scalar const&, LhsTensor&, RhsTensor&)
{
  if constexpr (detail::cpu_gemm_types_compatible<OutputTensor, Scalar, LhsTensor, RhsTensor>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Construct or resize a deferred host output and run reference GEMM.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::Scalar Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::HostGemmTensorAccess<OutputTensor, LhsTensor, RhsTensor>
KernelAttempt try_kernel(CpuReferenceBackend, assign_product_op const&,
                         async::shared_storage<OutputTensor>& output_storage, Scalar alpha, LhsTensor const& lhs,
                         RhsTensor const& rhs)
{
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  auto const shape = detail::matrix_product_shape(lhs_span, rhs_span);
  auto& output = uni20::prepare_output(output_storage, shape);
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  return detail::try_cpu_gemm(output_span, alpha, lhs_span, rhs_span, Scalar{});
}

} // namespace uni20::linalg
