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

namespace uni20::linalg
{
namespace detail
{
template <class OutputTensor, class LhsTensor, class RhsTensor>
concept HostGemmTensorAccess =
    uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
    uni20::detail::HostReadableTensor<RhsTensor>;

template <class OutputTensor, class Scalar, class LhsTensor, class RhsTensor> consteval bool cpu_gemm_types_compatible()
{
  using output_span = uni20::detail::host_write_tensor_mdspan_t<OutputTensor>;
  using lhs_span = uni20::detail::host_read_tensor_mdspan_t<LhsTensor>;
  using rhs_span = uni20::detail::host_read_tensor_mdspan_t<RhsTensor>;
  return uni20::linalg::cpu::GemmCompatible<output_span, Scalar, lhs_span, rhs_span>;
}
} // namespace detail

/// \brief Report eligibility for host DeviceTensorView CPU GEMM.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::Scalar Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::HostGemmTensorAccess<OutputTensor, LhsTensor, RhsTensor>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, gemm_op const&, OutputTensor&, Scalar const&,
                                    LhsTensor&, RhsTensor&, Scalar const&)
{
  if constexpr (detail::cpu_gemm_types_compatible<OutputTensor, Scalar, LhsTensor, RhsTensor>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host tensor access and run reference GEMM.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::Scalar Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::HostGemmTensorAccess<OutputTensor, LhsTensor, RhsTensor>
KernelAttempt try_kernel(CpuReferenceBackend, gemm_op const&, OutputTensor& output, Scalar alpha, LhsTensor const& lhs,
                         RhsTensor const& rhs, Scalar beta)
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
KernelAttempt try_kernel(CpuReferenceBackend backend, assign_product_op const&, OutputTensor& output, Scalar alpha,
                         LhsTensor const& lhs, RhsTensor const& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  uni20::prepare_output(output, shape);
  return try_kernel(backend, gemm_op{}, output, alpha, lhs, rhs, Scalar{});
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
KernelAttempt try_kernel(CpuReferenceBackend backend, assign_product_op const&,
                         async::shared_storage<OutputTensor>& output_storage, Scalar alpha, LhsTensor const& lhs,
                         RhsTensor const& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  auto& output = uni20::prepare_output(output_storage, shape);
  return try_kernel(backend, gemm_op{}, output, alpha, lhs, rhs, Scalar{});
}

} // namespace uni20::linalg
