#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief Reference CPU GEMV backend for tensor-view operands.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/cpu/gemv.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

namespace uni20::linalg
{
namespace detail
{
template <class OutputMdspan, class MatrixMdspan, class InputMdspan>
concept HostGemvMdspanAccess = uni20::HostWritableMdspec<OutputMdspan> && uni20::HostReadableMdspec<MatrixMdspan> &&
                               uni20::HostReadableMdspec<InputMdspan>;

template <class OutputMdspan, class Scalar, class MatrixMdspan, class InputMdspan>
consteval bool cpu_gemv_types_compatible()
{
  using output_span = uni20::host_write_mdspan_t<OutputMdspan>;
  using matrix_span = uni20::host_read_mdspan_t<MatrixMdspan>;
  using input_span = uni20::host_read_mdspan_t<InputMdspan>;
  return uni20::linalg::cpu::GemvCompatible<output_span, Scalar, matrix_span, input_span>;
}
} // namespace detail

/// \brief Report eligibility for host-accessible mdspec CPU GEMV.
template <uni20::MutableRankedMdspecLike<1> OutputMdspan, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> MatrixMdspan,
          uni20::RankedMdspecLike<1> InputMdspan>
  requires detail::HostGemvMdspanAccess<OutputMdspan, MatrixMdspan, InputMdspan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, gemv_op const&, OutputMdspan&, Scalar const&,
                                    MatrixMdspan&, InputMdspan&, Scalar const&)
{
  if constexpr (detail::cpu_gemv_types_compatible<OutputMdspan, Scalar, MatrixMdspan, InputMdspan>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host access and run reference GEMV.
template <uni20::MutableRankedMdspecLike<1> OutputMdspan, uni20::Scalar Scalar, uni20::RankedMdspecLike<2> MatrixMdspan,
          uni20::RankedMdspecLike<1> InputMdspan>
  requires detail::HostGemvMdspanAccess<OutputMdspan, MatrixMdspan, InputMdspan>
KernelAttempt try_kernel(CpuReferenceBackend, gemv_op const&, OutputMdspan& output, Scalar alpha, MatrixMdspan& matrix,
                         InputMdspan& input, Scalar beta)
{
  auto output_access = acquire_host_write_access_sync(output);
  auto matrix_access = acquire_host_read_access_sync(matrix);
  auto input_access = acquire_host_read_access_sync(input);
  auto output_span = output_access.mdspan();
  auto matrix_span = matrix_access.mdspan();
  auto input_span = input_access.mdspan();
  uni20::linalg::cpu::gemv(output_span, alpha, matrix_span, input_span, beta);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
