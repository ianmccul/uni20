#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief BLAS backend adapter for operation-tag GEMV dispatch.
 */

#include <uni20/linalg/blas/gemv.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <utility>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for direct BLAS GEMV dispatch.
template <uni20::MutableRankedStridedMdspanLike<1> OutputMdspan, class Scalar,
          uni20::RankedStridedMdspanLike<2> MatrixMdspan, uni20::RankedStridedMdspanLike<1> InputMdspan>
consteval auto kernel_accepts_types(BlasBackend const&, gemv_op const&, OutputMdspan&, Scalar const&, MatrixMdspan&,
                                    InputMdspan&, Scalar const&)
{
  if constexpr (requires(OutputMdspan& output, Scalar alpha, MatrixMdspan& matrix, InputMdspan& input) {
                  { uni20::linalg::blas::try_gemv(output, alpha, matrix, input, alpha) } -> std::same_as<KernelAttempt>;
                })
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Try GEMV through the direct mdspan BLAS wrapper.
template <class OutputMdspan, class Scalar, class MatrixMdspan, class InputMdspan>
KernelAttempt try_kernel(BlasBackend, gemv_op const&, OutputMdspan&& output, Scalar alpha, MatrixMdspan&& matrix,
                         InputMdspan&& input, Scalar beta)
{
  return uni20::linalg::blas::try_gemv(std::forward<OutputMdspan>(output), alpha, std::forward<MatrixMdspan>(matrix),
                                       std::forward<InputMdspan>(input), beta);
}

/// \brief Report eligibility for blocking DeviceTensorView BLAS GEMV.
template <uni20::MutableRankedDeviceTensorView<1> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> MatrixTensor, uni20::RankedDeviceTensorView<1> InputTensor>
  requires uni20::detail::BlockingWritableTensor<OutputTensor> && uni20::detail::BlockingReadableTensor<MatrixTensor> &&
           uni20::detail::BlockingReadableTensor<InputTensor>
consteval auto kernel_accepts_types(BlasBackend const&, gemv_op const&, OutputTensor&, Scalar const&, MatrixTensor&,
                                    InputTensor&, Scalar const&)
{
  using output_span = uni20::detail::blocking_write_tensor_mdspan_t<OutputTensor>;
  using matrix_span = uni20::detail::blocking_read_tensor_mdspan_t<MatrixTensor>;
  using input_span = uni20::detail::blocking_read_tensor_mdspan_t<InputTensor>;
  constexpr auto acceptance = detail::backend_type_acceptance<BlasBackend, gemv_op, output_span&, Scalar const&,
                                                              matrix_span&, input_span&, Scalar const&>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve blocking tensor access and try BLAS GEMV.
template <uni20::MutableRankedDeviceTensorView<1> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> MatrixTensor, uni20::RankedDeviceTensorView<1> InputTensor>
  requires uni20::detail::BlockingWritableTensor<OutputTensor> && uni20::detail::BlockingReadableTensor<MatrixTensor> &&
           uni20::detail::BlockingReadableTensor<InputTensor>
KernelAttempt try_kernel(BlasBackend backend, gemv_op const& op, OutputTensor& output, Scalar alpha,
                         MatrixTensor const& matrix, InputTensor const& input, Scalar beta)
{
  auto output_access = blocking_write_access(output);
  auto matrix_access = blocking_read_access(matrix);
  auto input_access = blocking_read_access(input);
  auto output_span = output_access.mdspan();
  auto matrix_span = matrix_access.mdspan();
  auto input_span = input_access.mdspan();
  return try_kernel(backend, op, output_span, alpha, matrix_span, input_span, beta);
}

} // namespace uni20::linalg
