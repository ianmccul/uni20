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

template <class OutputMdspan, class Scalar, class MatrixMdspan, class InputMdspan>
consteval bool gemv_types_compatible()
{
  using output_span = uni20::host_write_mdspan_t<OutputMdspan>;
  using matrix_span = uni20::host_read_mdspan_t<MatrixMdspan>;
  using input_span = uni20::host_read_mdspan_t<InputMdspan>;
  return requires(output_span& output, Scalar alpha, matrix_span& matrix, input_span& input) {
    { uni20::linalg::blas::try_gemv(output, alpha, matrix, input, alpha) } -> std::same_as<KernelAttempt>;
  };
}
} // namespace detail::blas_backend

/// \brief Report eligibility for host-accessible mdspec BLAS GEMV.
template <uni20::MutableRankedMdspecLike<1> OutputMdspan, class Scalar, uni20::RankedMdspecLike<2> MatrixMdspan,
          uni20::RankedMdspecLike<1> InputMdspan>
  requires uni20::HostWritableMdspec<OutputMdspan> && uni20::HostReadableMdspec<MatrixMdspan> &&
           uni20::HostReadableMdspec<InputMdspan>
consteval auto kernel_accepts_types(BlasBackend const&, gemv_op const&, OutputMdspan&, Scalar const&, MatrixMdspan&,
                                    InputMdspan&, Scalar const&)
{
  if constexpr (detail::blas_backend::gemv_types_compatible<OutputMdspan, Scalar, MatrixMdspan, InputMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Resolve host access and try BLAS GEMV.
template <uni20::MutableRankedMdspecLike<1> OutputMdspan, class Scalar, uni20::RankedMdspecLike<2> MatrixMdspan,
          uni20::RankedMdspecLike<1> InputMdspan>
  requires uni20::HostWritableMdspec<OutputMdspan> && uni20::HostReadableMdspec<MatrixMdspan> &&
           uni20::HostReadableMdspec<InputMdspan>
KernelAttempt try_kernel(BlasBackend, gemv_op const&, OutputMdspan& output, Scalar alpha, MatrixMdspan& matrix,
                         InputMdspan& input, Scalar beta)
{
  auto output_access = acquire_host_write_access_sync(output);
  auto matrix_access = acquire_host_read_access_sync(matrix);
  auto input_access = acquire_host_read_access_sync(input);
  auto output_span = output_access.mdspan();
  auto matrix_span = matrix_access.mdspan();
  auto input_span = input_access.mdspan();
  return uni20::linalg::blas::try_gemv(output_span, alpha, matrix_span, input_span, beta);
}

} // namespace uni20::linalg
