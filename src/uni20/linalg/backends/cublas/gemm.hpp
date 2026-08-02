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

template <uni20::MdspecLike Mdspan> [[nodiscard]] uni20::cuda::Device span_device(Mdspan const& span)
{
  return blas::detail::span_data(span).buffer().device();
}

[[nodiscard]] inline bool product_output_is_empty(detail::matrix_product_extents const& shape) noexcept
{
  return shape.extent(0) == 0 || shape.extent(1) == 0;
}

} // namespace detail::cublas_backend

/// \brief Report cuBLAS eligibility for normalized mdspec GEMM operands.
template <uni20::MutableRankedMdspecLike<2> OutputMdspan, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
consteval auto kernel_accepts_types(CublasBackend const&, gemm_op const&, OutputMdspan&, Scalar const&, LhsMdspan&,
                                    RhsMdspan&, Scalar const&)
{
  if constexpr (detail::cublas_backend::accepts_gemm_types<Scalar, OutputMdspan, LhsMdspan, RhsMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Invoke the cuBLAS GEMM adapter with normalized mdspecs.
template <uni20::MutableRankedMdspecLike<2> OutputMdspan, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
KernelAttempt try_kernel(CublasBackend, gemm_op const&, OutputMdspan& output, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs, Scalar beta)
{
  return detail::cublas_backend::try_gemm(output, alpha, lhs, rhs, beta);
}

/// \brief Report cuBLAS eligibility for replaceable-output matrix products.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
consteval auto kernel_accepts_types(CublasBackend const&, assign_product_op const&, OutputTensor&, Scalar const&,
                                    LhsMdspan&, RhsMdspan&)
{
  using output_span = std::remove_cvref_t<decltype(uni20::mdspec_of(std::declval<OutputTensor&>()))>;
  if constexpr (detail::cublas_backend::accepts_gemm_types<Scalar, output_span, LhsMdspan, RhsMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Prepare a replaceable output and lower normalized operands for cuBLAS GEMM.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
KernelAttempt try_kernel(CublasBackend, assign_product_op const&, OutputTensor& output, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  bool const output_is_empty = detail::cublas_backend::product_output_is_empty(shape);
  auto const lhs_device = detail::cublas_backend::span_device(lhs);
  if (!output_is_empty && detail::cublas_backend::span_device(rhs) != lhs_device)
    return KernelAttempt::incompatible_devices;

  if constexpr (requires { uni20::prepare_output(output, shape, lhs_device); })
  {
    uni20::prepare_output(output, shape, lhs_device);
  }
  else if (!uni20::tensor_extents_equal(output.extents(), shape))
  {
    return KernelAttempt::unsupported_shape;
  }

  auto output_span = uni20::mdspec_of(output);
  if (!output_is_empty && detail::cublas_backend::span_device(output_span) != lhs_device)
    return KernelAttempt::incompatible_devices;
  return detail::cublas_backend::try_gemm(output_span, alpha, lhs, rhs, Scalar{});
}

/// \brief Report cuBLAS eligibility for a deferred Tensor matrix-product output.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires requires(async::shared_storage<OutputTensor>& storage, detail::matrix_product_extents const& shape,
                    uni20::cuda::Device device) { uni20::prepare_output(storage, shape, device); }
consteval auto kernel_accepts_types(CublasBackend const&, assign_product_op const&,
                                    async::shared_storage<OutputTensor>&, Scalar const&, LhsMdspan&, RhsMdspan&)
{
  using output_span = std::remove_cvref_t<decltype(uni20::mdspec_of(std::declval<OutputTensor&>()))>;
  if constexpr (detail::cublas_backend::accepts_gemm_types<Scalar, output_span, LhsMdspan, RhsMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Construct or relocate a deferred CUDA output and run blocking cuBLAS GEMM.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires requires(async::shared_storage<OutputTensor>& storage, detail::matrix_product_extents const& shape,
                    uni20::cuda::Device device) { uni20::prepare_output(storage, shape, device); }
KernelAttempt try_kernel(CublasBackend, assign_product_op const&, async::shared_storage<OutputTensor>& output_storage,
                         Scalar alpha, LhsMdspan& lhs, RhsMdspan& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  bool const output_is_empty = detail::cublas_backend::product_output_is_empty(shape);
  auto const lhs_device = detail::cublas_backend::span_device(lhs);
  if (!output_is_empty && detail::cublas_backend::span_device(rhs) != lhs_device)
    return KernelAttempt::incompatible_devices;

  auto& output = uni20::prepare_output(output_storage, shape, lhs_device);
  auto output_span = uni20::mdspec_of(output);
  return detail::cublas_backend::try_gemm(output_span, alpha, lhs, rhs, Scalar{});
}

} // namespace uni20::linalg
