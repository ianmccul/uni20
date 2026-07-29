#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief cuBLAS backend adapter for provider-ready GEMM dispatch.
 */

#include <uni20/linalg/backends/cublas/detail/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/matrix_product_shape.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/output.hpp>

#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cublas_backend
{

template <uni20::DeviceMdspanLike Mdspan> [[nodiscard]] uni20::cuda::Device span_device(Mdspan const& span)
{
  return blas::detail::span_data(span).buffer().device();
}

} // namespace detail::cublas_backend

/// \brief Report cuBLAS eligibility for normalized device-mdspan GEMM operands.
template <uni20::MutableRankedDeviceMdspanLike<2> OutputMdspan, class Scalar,
          uni20::RankedDeviceMdspanLike<2> LhsMdspan, uni20::RankedDeviceMdspanLike<2> RhsMdspan>
consteval auto kernel_accepts_types(CublasBackend const&, gemm_op const&, OutputMdspan&, Scalar const&, LhsMdspan&,
                                    RhsMdspan&, Scalar const&)
{
  if constexpr (detail::cublas_backend::accepts_gemm_types<Scalar, OutputMdspan, LhsMdspan, RhsMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Invoke the cuBLAS GEMM adapter with normalized device mdspans.
template <uni20::MutableRankedDeviceMdspanLike<2> OutputMdspan, class Scalar,
          uni20::RankedDeviceMdspanLike<2> LhsMdspan, uni20::RankedDeviceMdspanLike<2> RhsMdspan>
KernelAttempt try_kernel(CublasBackend, gemm_op const&, OutputMdspan& output, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs, Scalar beta)
{
  return detail::cublas_backend::try_gemm(output, alpha, lhs, rhs, beta);
}

/// \brief Report cuBLAS eligibility for replaceable-output matrix products.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
consteval auto kernel_accepts_types(CublasBackend const&, assign_product_op const&, OutputTensor&, Scalar const&,
                                    LhsTensor&, RhsTensor&)
{
  using output_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<OutputTensor&>()))>;
  using lhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<LhsTensor const&>()))>;
  using rhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<RhsTensor const&>()))>;
  if constexpr (detail::cublas_backend::accepts_gemm_types<Scalar, output_span, lhs_span, rhs_span>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Prepare a replaceable output and lower the operands once for cuBLAS GEMM.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
KernelAttempt try_kernel(CublasBackend, assign_product_op const&, OutputTensor& output, Scalar alpha,
                         LhsTensor const& lhs, RhsTensor const& rhs)
{
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  auto const shape = detail::matrix_product_shape(lhs_span, rhs_span);
  auto const lhs_device = detail::cublas_backend::span_device(lhs_span);
  if (detail::cublas_backend::span_device(rhs_span) != lhs_device) return KernelAttempt::incompatible_devices;

  if constexpr (requires { uni20::prepare_output(output, shape, lhs_device); })
  {
    uni20::prepare_output(output, shape, lhs_device);
  }
  else if (!uni20::detail::tensor_extents_equal(output.extents(), shape))
  {
    return KernelAttempt::unsupported_shape;
  }

  auto output_span = uni20::detail::tensor_device_mdspan(output);
  if (detail::cublas_backend::span_device(output_span) != lhs_device) return KernelAttempt::incompatible_devices;
  return detail::cublas_backend::try_gemm(output_span, alpha, lhs_span, rhs_span, Scalar{});
}

/// \brief Report cuBLAS eligibility for a deferred Tensor matrix-product output.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires requires(async::shared_storage<OutputTensor>& storage, detail::matrix_product_extents const& shape,
                    uni20::cuda::Device device) { uni20::prepare_output(storage, shape, device); }
consteval auto kernel_accepts_types(CublasBackend const&, assign_product_op const&,
                                    async::shared_storage<OutputTensor>&, Scalar const&, LhsTensor&, RhsTensor&)
{
  using output_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<OutputTensor&>()))>;
  using lhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<LhsTensor const&>()))>;
  using rhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<RhsTensor const&>()))>;
  if constexpr (detail::cublas_backend::accepts_gemm_types<Scalar, output_span, lhs_span, rhs_span>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Construct or relocate a deferred CUDA output and run blocking cuBLAS GEMM.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires requires(async::shared_storage<OutputTensor>& storage, detail::matrix_product_extents const& shape,
                    uni20::cuda::Device device) { uni20::prepare_output(storage, shape, device); }
KernelAttempt try_kernel(CublasBackend, assign_product_op const&, async::shared_storage<OutputTensor>& output_storage,
                         Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs)
{
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  auto const shape = detail::matrix_product_shape(lhs_span, rhs_span);
  auto const lhs_device = detail::cublas_backend::span_device(lhs_span);
  if (detail::cublas_backend::span_device(rhs_span) != lhs_device) return KernelAttempt::incompatible_devices;

  auto& output = uni20::prepare_output(output_storage, shape, lhs_device);
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  return detail::cublas_backend::try_gemm(output_span, alpha, lhs_span, rhs_span, Scalar{});
}

} // namespace uni20::linalg
