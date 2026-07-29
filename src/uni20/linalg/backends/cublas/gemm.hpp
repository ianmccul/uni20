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

template <uni20::DeviceTensorView Tensor> [[nodiscard]] uni20::cuda::Device tensor_device(Tensor const& tensor)
{
  auto span = uni20::detail::tensor_device_mdspan(tensor);
  return blas::detail::span_data(span).buffer().device();
}

template <uni20::DeviceTensorView OutputTensor>
[[nodiscard]] bool output_shape_matches(OutputTensor const& output, matrix_product_extents const& shape)
{
  return uni20::detail::tensor_extents_equal(output.extents(), shape);
}

} // namespace detail::cublas_backend

/// \brief Report cuBLAS eligibility for DeviceTensorView GEMM operands.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
consteval auto kernel_accepts_types(CublasBackend const&, gemm_op const&, OutputTensor&, Scalar const&, LhsTensor&,
                                    RhsTensor&, Scalar const&)
{
  using output_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<OutputTensor&>()))>;
  using lhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<LhsTensor const&>()))>;
  using rhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<RhsTensor const&>()))>;
  if constexpr (detail::cublas_backend::accepts_gemm_types<Scalar, output_span, lhs_span, rhs_span>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Lower DeviceTensorView operands and invoke the cuBLAS GEMM adapter.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
KernelAttempt try_kernel(CublasBackend, gemm_op const&, OutputTensor& output, Scalar alpha, LhsTensor const& lhs,
                         RhsTensor const& rhs, Scalar beta)
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  return detail::cublas_backend::try_gemm(output_span, alpha, lhs_span, rhs_span, beta);
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

/// \brief Lower a replaceable-output matrix product to fixed-output cuBLAS GEMM.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
KernelAttempt try_kernel(CublasBackend backend, assign_product_op const&, OutputTensor& output, Scalar alpha,
                         LhsTensor const& lhs, RhsTensor const& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  auto const lhs_device = detail::cublas_backend::tensor_device(lhs);
  if (detail::cublas_backend::tensor_device(rhs) != lhs_device) return KernelAttempt::incompatible_devices;

  bool const shape_matches = detail::cublas_backend::output_shape_matches(output, shape);
  bool const device_matches = detail::cublas_backend::tensor_device(output) == lhs_device;
  bool const replacement_required = !shape_matches || !device_matches;
  if (replacement_required)
  {
    if constexpr (requires { uni20::prepare_output(output, shape, lhs_device); })
    {
      uni20::prepare_output(output, shape, lhs_device);
    }
    else
    {
      return shape_matches ? KernelAttempt::incompatible_devices : KernelAttempt::unsupported_shape;
    }
  }

  KernelAttempt const attempt = try_kernel(backend, gemm_op{}, output, alpha, lhs, rhs, Scalar{});
  return attempt;
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
KernelAttempt try_kernel(CublasBackend backend, assign_product_op const&,
                         async::shared_storage<OutputTensor>& output_storage, Scalar alpha, LhsTensor const& lhs,
                         RhsTensor const& rhs)
{
  if (output_storage.constructed()) return try_kernel(backend, assign_product_op{}, *output_storage, alpha, lhs, rhs);

  auto const shape = detail::matrix_product_shape(lhs, rhs);
  auto const lhs_device = detail::cublas_backend::tensor_device(lhs);
  if (detail::cublas_backend::tensor_device(rhs) != lhs_device) return KernelAttempt::incompatible_devices;

  auto& output = uni20::prepare_output(output_storage, shape, lhs_device);
  KernelAttempt const attempt = try_kernel(backend, gemm_op{}, output, alpha, lhs, rhs, Scalar{});
  return attempt;
}

} // namespace uni20::linalg
