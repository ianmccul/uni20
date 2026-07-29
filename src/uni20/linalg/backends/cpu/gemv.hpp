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
template <class OutputTensor, class MatrixTensor, class InputTensor>
concept HostGemvTensorAccess =
    uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<MatrixTensor> &&
    uni20::detail::HostReadableTensor<InputTensor>;

template <class OutputTensor, class Scalar, class MatrixTensor, class InputTensor>
consteval bool cpu_gemv_types_compatible()
{
  using output_span = uni20::detail::host_write_tensor_mdspan_t<OutputTensor>;
  using matrix_span = uni20::detail::host_read_tensor_mdspan_t<MatrixTensor>;
  using input_span = uni20::detail::host_read_tensor_mdspan_t<InputTensor>;
  return uni20::linalg::cpu::GemvCompatible<output_span, Scalar, matrix_span, input_span>;
}
} // namespace detail

/// \brief Report eligibility for host DeviceTensorView CPU GEMV.
template <uni20::MutableRankedDeviceTensorView<1> OutputTensor, uni20::Scalar Scalar,
          uni20::RankedDeviceTensorView<2> MatrixTensor, uni20::RankedDeviceTensorView<1> InputTensor>
  requires detail::HostGemvTensorAccess<OutputTensor, MatrixTensor, InputTensor>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, gemv_op const&, OutputTensor&, Scalar const&,
                                    MatrixTensor&, InputTensor&, Scalar const&)
{
  if constexpr (detail::cpu_gemv_types_compatible<OutputTensor, Scalar, MatrixTensor, InputTensor>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host tensor access and run reference GEMV.
template <uni20::MutableRankedDeviceTensorView<1> OutputTensor, uni20::Scalar Scalar,
          uni20::RankedDeviceTensorView<2> MatrixTensor, uni20::RankedDeviceTensorView<1> InputTensor>
  requires detail::HostGemvTensorAccess<OutputTensor, MatrixTensor, InputTensor>
KernelAttempt try_kernel(CpuReferenceBackend, gemv_op const&, OutputTensor& output, Scalar alpha,
                         MatrixTensor const& matrix, InputTensor const& input, Scalar beta)
{
  auto output_access = acquire_host_write_access(output);
  auto matrix_access = acquire_host_read_access(matrix);
  auto input_access = acquire_host_read_access(input);
  auto output_span = output_access.mdspan();
  auto matrix_span = matrix_access.mdspan();
  auto input_span = input_access.mdspan();
  uni20::linalg::cpu::gemv(output_span, alpha, matrix_span, input_span, beta);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
