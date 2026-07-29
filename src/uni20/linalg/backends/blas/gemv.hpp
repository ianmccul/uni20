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

namespace uni20::linalg
{
namespace detail::blas_backend
{

template <class OutputTensor, class Scalar, class MatrixTensor, class InputTensor>
consteval bool gemv_types_compatible()
{
  using output_span = uni20::detail::host_write_tensor_mdspan_t<OutputTensor>;
  using matrix_span = uni20::detail::host_read_tensor_mdspan_t<MatrixTensor>;
  using input_span = uni20::detail::host_read_tensor_mdspan_t<InputTensor>;
  return requires(output_span& output, Scalar alpha, matrix_span& matrix, input_span& input) {
    { uni20::linalg::blas::try_gemv(output, alpha, matrix, input, alpha) } -> std::same_as<KernelAttempt>;
  };
}
} // namespace detail::blas_backend

/// \brief Report eligibility for host DeviceTensorView BLAS GEMV.
template <uni20::MutableRankedDeviceTensorView<1> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> MatrixTensor, uni20::RankedDeviceTensorView<1> InputTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<MatrixTensor> &&
           uni20::detail::HostReadableTensor<InputTensor>
consteval auto kernel_accepts_types(BlasBackend const&, gemv_op const&, OutputTensor&, Scalar const&, MatrixTensor&,
                                    InputTensor&, Scalar const&)
{
  if constexpr (detail::blas_backend::gemv_types_compatible<OutputTensor, Scalar, MatrixTensor, InputTensor>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Resolve host tensor access and try BLAS GEMV.
template <uni20::MutableRankedDeviceTensorView<1> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> MatrixTensor, uni20::RankedDeviceTensorView<1> InputTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<MatrixTensor> &&
           uni20::detail::HostReadableTensor<InputTensor>
KernelAttempt try_kernel(BlasBackend, gemv_op const&, OutputTensor& output, Scalar alpha, MatrixTensor const& matrix,
                         InputTensor const& input, Scalar beta)
{
  auto output_access = acquire_host_write_access(output);
  auto matrix_access = acquire_host_read_access(matrix);
  auto input_access = acquire_host_read_access(input);
  auto output_span = output_access.mdspan();
  auto matrix_span = matrix_access.mdspan();
  auto input_span = input_access.mdspan();
  return uni20::linalg::blas::try_gemv(output_span, alpha, matrix_span, input_span, beta);
}

} // namespace uni20::linalg
