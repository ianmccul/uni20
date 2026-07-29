#pragma once

/**
 * \file matrix_product.hpp
 * \ingroup linalg
 * \brief Tensor-level matrix product update and overwrite operations.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/matrix_product_shape.hpp>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/tensor/output.hpp>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class OutputTensor, class InputTensor>
[[nodiscard]] constexpr bool is_obvious_tensor_alias(OutputTensor& output, InputTensor const& input) noexcept
{
  if constexpr (std::same_as<std::remove_cvref_t<OutputTensor>, std::remove_cvref_t<InputTensor>>)
  {
    return static_cast<void const*>(std::addressof(output)) == static_cast<void const*>(std::addressof(input));
  }
  return false;
}

template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::RankedDeviceTensorView<2> LhsTensor,
          uni20::RankedDeviceTensorView<2> RhsTensor>
void validate_matrix_product_aliasing(OutputTensor& output, LhsTensor const& lhs, RhsTensor const& rhs)
{
  ERROR_IF(is_obvious_tensor_alias(output, lhs) || is_obvious_tensor_alias(output, rhs),
           "matrix product output must not alias an input tensor");
}

template <class OutputTensor, class LhsTensor, class RhsTensor>
concept CompatibleMatrixProductTensors =
    std::same_as<uni20::tensor_element_t<OutputTensor>, uni20::tensor_element_t<LhsTensor>> &&
    std::same_as<uni20::tensor_element_t<OutputTensor>, uni20::tensor_element_t<RhsTensor>>;
} // namespace detail

/// \brief Accumulate a matrix product into a fixed-shape Tensor output.
/// \details Computes `output += alpha * lhs * rhs`. The output shape is
///          validated and never resized because its old values participate in
///          the result.
/// \pre Output storage does not overlap either input, beyond the cheap
///      same-object aliases rejected by this front end.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor>
void add_product(BackendSelector&& selector, OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs,
                 uni20::tensor_element_t<OutputTensor> alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  detail::validate_matrix_product_aliasing(output, lhs, rhs);
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  uni20::require_output(output, shape);
  gemm(std::forward<BackendSelector>(selector), std::forward<OutputTensor>(output), alpha, lhs, rhs,
       uni20::tensor_element_t<OutputTensor>{1});
}

/// \brief Accumulate a matrix product using the operands' default backend selector.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::RankedDeviceTensorView<2> LhsTensor,
          uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor>
void add_product(OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs,
                 uni20::tensor_element_t<OutputTensor> alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  detail::validate_matrix_product_aliasing(output, lhs, rhs);
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  uni20::require_output(output, shape);
  auto selector = select_backend(gemm_op{}, output, lhs, rhs);
  gemm(selector, std::forward<OutputTensor>(output), alpha, lhs, rhs, uni20::tensor_element_t<OutputTensor>{1});
}

/// \brief Overwrite a resizable or already-compatible Tensor with a matrix product.
/// \details Computes `output = alpha * lhs * rhs`. Resizable outputs are
///          prepared before their writable mdspan is resolved; fixed outputs
///          must already have the required shape.
/// \pre Output storage does not overlap either input, beyond the cheap
///      same-object aliases rejected by this front end.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor>
void assign_product(BackendSelector&& selector, OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs,
                    uni20::tensor_element_t<OutputTensor> alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  detail::validate_matrix_product_aliasing(output, lhs, rhs);
  dispatch_kernel(std::forward<BackendSelector>(selector), assign_product_op{}, output, alpha, lhs, rhs);
}

/// \brief Overwrite a Tensor with a matrix product using its default backend selector.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::RankedDeviceTensorView<2> LhsTensor,
          uni20::RankedDeviceTensorView<2> RhsTensor>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor>
void assign_product(OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs,
                    uni20::tensor_element_t<OutputTensor> alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  detail::validate_matrix_product_aliasing(output, lhs, rhs);
  auto selector = select_backend(assign_product_op{}, output, lhs, rhs);
  dispatch_kernel(selector, assign_product_op{}, output, alpha, lhs, rhs);
}

} // namespace uni20::linalg
